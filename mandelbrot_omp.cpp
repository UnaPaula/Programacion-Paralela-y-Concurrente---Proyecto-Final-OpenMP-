/*
 * ============================================================
 *  Mandelbrot 8K + Filtro de Convolución 2D — versión OpenMP
 * ============================================================
 *
 *  Compilar:
 *    g++ -O2 -std=c++17 -fopenmp -o mandelbrot_omp mandelbrot_omp.cpp
 *
 *  Controlar hilos desde el shell (opcional):
 *    export OMP_NUM_THREADS=8
 *    ./mandelbrot_omp
 *
 *  Genera dos archivos PPM:
 *    mandelbrot_8k.ppm          — fractal original
 *    mandelbrot_8k_filtro.ppm   — desenfoque Gaussiano + realce Sobel
 *
 *  ──────────────────────────────────────────────────────────
 *  ESTRATEGIA DE PARALELIZACIÓN (OpenMP)
 *  ──────────────────────────────────────────────────────────
 *
 *  1. generate_mandelbrot()
 *     #pragma omp parallel for schedule(static)
 *     El conjunto de Mandelbrot tiene carga muy desigual por fila
 *     (el interior itera MAX_ITER veces, el exterior pocas).
 *     schedule(dynamic) reparte filas bajo demanda → equilibrio
 *     de carga óptimo sin trabajo desperdiciado.
 *     Cada hilo escribe en su propia franja de img (sin overlap),
 *     por lo que NO se necesita ninguna sección crítica.
 *
 *  2. gaussian_blur_separable()
 *     Pasada horizontal y pasada vertical son completamente
 *     independientes. Cada pasada usa:
 *     #pragma omp parallel for schedule(static)
 *     porque la carga ES uniforme (mismo kernel por pixel).
 *     schedule(static) minimiza overhead de scheduling.
 *     Los buffers tmp* y out* se escriben en posiciones
 *     disjuntas por fila → cero condiciones de carrera.
 *
 *  3. sobel_edge_highlight()
 *     Tres bucles independientes:
 *     a) Conversión a gris       → parallel for schedule(static)
 *     b) Magnitud Sobel          → parallel for schedule(static)
 *        + reducción de máximo:  reduction(max : maxEdge)
 *        OpenMP garantiza la reducción atómica correctamente.
 *     c) Mezcla final            → parallel for schedule(static)
 *
 *  4. print_stats()
 *     Suma de canales con reduction(+: sr, sg, sb)
 * ============================================================
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <string>
#include <omp.h>
#include <cstdint>
#include <unordered_map>

using namespace std::chrono;

// ─────────────────────────────────────────────
//  Dimensiones (8K = 7680×4320)
// ─────────────────────────────────────────────
static constexpr int WIDTH    = 7680;
static constexpr int HEIGHT   = 4320;

// ─────────────────────────────────────────────
//  Parámetros del Mandelbrot
// ─────────────────────────────────────────────
static constexpr int    MAX_ITER = 1024;
static constexpr double CENTER_X = -0.7435669;
static constexpr double CENTER_Y =  0.1314023;
static constexpr double ZOOM     =  0.00045;

// ─────────────────────────────────────────────
//  Tipos auxiliares
// ─────────────────────────────────────────────
struct RGB { uint8_t r, g, b; };
using Image = std::vector<RGB>;

// ─────────────────────────────────────────────
//  Cronómetro con omp_get_wtime (wall-clock real)
// ─────────────────────────────────────────────
class Timer {
    high_resolution_clock::time_point t0;
public:
    void   start()           { t0 = high_resolution_clock::now(); }
    double elapsed_s() const { return duration<double>(high_resolution_clock::now() - t0).count(); }
};

// ─────────────────────────────────────────────
//  Barra de progreso (llamar solo desde 1 hilo)
// ─────────────────────────────────────────────
void progress_bar(const char* label, int done, int total, double elapsed) {
    static const int BAR = 40;
    double frac   = (double)done / total;
    int    filled = (int)(frac * BAR);
    std::cout << '\r' << label << " [";
    for (int i = 0; i < BAR; ++i)
        std::cout << (i < filled ? '#' : '-');
    double eta = (done > 0 && frac < 1.0) ? elapsed / frac * (1.0 - frac) : 0.0;
    std::cout << "] " << std::fixed << std::setprecision(1)
    << frac * 100.0 << "%  "
    << elapsed << "s  ETA " << eta << "s   " << std::flush;
    if (done == total) std::cout << '\n';
}

// ─────────────────────────────────────────────
//  Mapa de colores (pure function — thread-safe)
// ─────────────────────────────────────────────
static RGB iter_to_color(double smooth) {
    double t = std::fmod(smooth / MAX_ITER * 6.0, 1.0);
    struct Stop { double pos, r, g, b; };
    static const Stop stops[] = {
        { 0.00,   0,   0,  32 },
        { 0.20,   0,   7, 100 },
        { 0.40,  32, 107, 203 },
        { 0.55, 237, 255, 255 },
        { 0.70, 255, 170,   0 },
        { 0.85, 160,   0,   0 },
        { 1.00,   0,   0,  32 },
    };
    static const int N = 7;
    int lo = 0;
    for (int i = 0; i < N - 1; ++i)
        if (t >= stops[i].pos && t < stops[i+1].pos) { lo = i; break; }
        double range = stops[lo+1].pos - stops[lo].pos;
    double u = (range > 0.0) ? (t - stops[lo].pos) / range : 0.0;
    auto lerp = [&](double a, double b){ return a + u*(b-a); };
    return {
        (uint8_t)std::clamp(lerp(stops[lo].r, stops[lo+1].r), 0.0, 255.0),
        (uint8_t)std::clamp(lerp(stops[lo].g, stops[lo+1].g), 0.0, 255.0),
        (uint8_t)std::clamp(lerp(stops[lo].b, stops[lo+1].b), 0.0, 255.0)
    };
}

// ══════════════════════════════════════════════
//  PASO 1 — Generación del fractal Mandelbrot
//
//  schedule(dynamic, 4): la carga por fila varía
//  mucho (interior = MAX_ITER iters, exterior = pocas).
//  Dynamic reparte chunks de 4 filas bajo demanda,
//  logrando balance de carga automático.
//  Sin race conditions: cada hilo escribe en filas
//  distintas del array img.
// ══════════════════════════════════════════════
Image generate_mandelbrot(omp_sched_t schd, int chunk) {
    Image img(WIDTH * HEIGHT);
    Timer t; t.start();
    const double asp = (double)WIDTH / HEIGHT;

    // Barra de progreso atómica (hilo 0 la actualiza)
    // Usamos una variable compartida para el último py visible

    omp_set_schedule(omp_sched_dynamic, chunk);

    #pragma omp parallel for schedule(runtime)
    for (int py = 0; py < HEIGHT; ++py) {
        for (int px = 0; px < WIDTH; ++px) {
            double cx = CENTER_X + (px / (double)WIDTH  - 0.5) * ZOOM * asp * 2.0;
            double cy = CENTER_Y + (py / (double)HEIGHT - 0.5) * ZOOM * 2.0;

            double zx = 0.0, zy = 0.0;
            int iter = 0;
            while (zx*zx + zy*zy < (1 << 16) && iter < MAX_ITER) {
                double tmp = zx*zx - zy*zy + cx;
                zy = 2.0*zx*zy + cy;
                zx = tmp;
                ++iter;
            }

            if (iter < MAX_ITER) {
                double log_zn = std::log(zx*zx + zy*zy) / 2.0;
                double nu     = std::log(log_zn / std::log(2.0)) / std::log(2.0);
                img[py * WIDTH + px] = iter_to_color(iter + 1 - nu);
            } else {
                img[py * WIDTH + px] = {0, 0, 0};
            }
        }
        // Solo el hilo 0 imprime progreso para evitar interleaving en cout
        if (omp_get_thread_num() == 0) {
            if (py % 50 == 0 || py == HEIGHT - 1)
                progress_bar("Generando Mandelbrot", py + 1, HEIGHT, t.elapsed_s());
        }
    }
    // Aseguramos que la barra llegue al 100%
    progress_bar("Generando Mandelbrot", HEIGHT, HEIGHT, t.elapsed_s());
    return img;
}

// ══════════════════════════════════════════════
//  PASO 2 — Desenfoque Gaussiano separable
//
//  Dos pasadas 1D (H luego V). Cada pasada es un
//  parallel for schedule(static) porque la carga
//  por píxel es uniforme (2*radius+1 MACs siempre).
//  Los buffers tmpR/G/B y outR/G/B son distintos →
//  cero solapamiento entre hilos.
//  radius y k1d.data() se capturan como const →
//  solo lectura, thread-safe sin copia.
// ══════════════════════════════════════════════
Image gaussian_blur_separable(const Image& src, int radius, float sigma) {
    const int ksize = 2 * radius + 1;

    // Kernel 1D (construido en serie, luego solo lectura)
    std::vector<float> k1d(ksize);
    {
        float s = 0;
        for (int i = -radius; i <= radius; ++i) {
            k1d[i + radius] = std::exp(-(float)(i*i) / (2.0f*sigma*sigma));
            s += k1d[i + radius];
        }
        for (auto& v : k1d) v /= s;
    }
    const float* kp = k1d.data();   // puntero const para el paralelo

    const int N = WIDTH * HEIGHT;
    std::vector<float> tmpR(N), tmpG(N), tmpB(N);
    std::vector<float> outR(N), outG(N), outB(N);

    Timer t; t.start();

    // ── Pasada horizontal ──
    #pragma omp parallel for schedule(static)
    for (int py = 0; py < HEIGHT; ++py) {
        for (int px = 0; px < WIDTH; ++px) {
            float r = 0, g = 0, b = 0;
            for (int k = -radius; k <= radius; ++k) {
                int sx = std::clamp(px + k, 0, WIDTH - 1);
                const RGB& pix = src[py * WIDTH + sx];
                float w = kp[k + radius];
                r += pix.r * w;  g += pix.g * w;  b += pix.b * w;
            }
            int idx = py * WIDTH + px;
            tmpR[idx] = r;  tmpG[idx] = g;  tmpB[idx] = b;
        }
        if (omp_get_thread_num() == 0 && (py % 100 == 0 || py == HEIGHT-1))
            progress_bar("Gauss horizontal   ", py + 1, HEIGHT, t.elapsed_s());
    }
    progress_bar("Gauss horizontal   ", HEIGHT, HEIGHT, t.elapsed_s());

    // ── Pasada vertical ──
    #pragma omp parallel for schedule(static)
    for (int py = 0; py < HEIGHT; ++py) {
        for (int px = 0; px < WIDTH; ++px) {
            float r = 0, g = 0, b = 0;
            for (int k = -radius; k <= radius; ++k) {
                int sy  = std::clamp(py + k, 0, HEIGHT - 1);
                int idx = sy * WIDTH + px;
                float w = kp[k + radius];
                r += tmpR[idx] * w;  g += tmpG[idx] * w;  b += tmpB[idx] * w;
            }
            int oi = py * WIDTH + px;
            outR[oi] = r;  outG[oi] = g;  outB[oi] = b;
        }
        if (omp_get_thread_num() == 0 && (py % 100 == 0 || py == HEIGHT-1))
            progress_bar("Gauss vertical     ", py + 1, HEIGHT, t.elapsed_s());
    }
    progress_bar("Gauss vertical     ", HEIGHT, HEIGHT, t.elapsed_s());

    // ── Convertir float → uint8 ──
    Image result(N);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        result[i] = {
            (uint8_t)std::clamp((int)outR[i], 0, 255),
            (uint8_t)std::clamp((int)outG[i], 0, 255),
            (uint8_t)std::clamp((int)outB[i], 0, 255)
        };
    return result;
}


// ══════════════════════════════════════════════
//  PASO 3 — Filtro Sobel + realce de bordes
//
//  a) Gris: parallel for trivial.
//  b) Sobel: parallel for + reduction(max:maxEdge).
//     Cada hilo calcula un maxEdge local; OpenMP los
//     combina automáticamente al final de la región.
//     No se necesita atomic ni critical.
//  c) Mezcla: parallel for trivial.
// ══════════════════════════════════════════════
Image sobel_edge_highlight(const Image& src, const Image& blurred) {
    // Kernels como arrays estáticos (solo lectura, thread-safe)
    static const int Gx[3][3] = {{ -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 }};
    static const int Gy[3][3] = {{ -1,-2,-1 }, {  0, 0, 0 }, {  1, 2, 1 }};

    const int N = WIDTH * HEIGHT;

    // a) Conversión a gris
    std::vector<float> gray(N);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        gray[i] = 0.2126f*src[i].r + 0.7152f*src[i].g + 0.0722f*src[i].b;

    // b) Magnitud Sobel + máximo global mediante reducción
    std::vector<float> edges(N, 0.0f);
    float maxEdge = 0.0f;
    Timer t; t.start();

    #pragma omp parallel for schedule(static) reduction(max : maxEdge)
    for (int py = 1; py < HEIGHT - 1; ++py) {
        for (int px = 1; px < WIDTH - 1; ++px) {
            float sx = 0, sy = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    float g = gray[(py+dy)*WIDTH + (px+dx)];
                    sx += g * Gx[dy+1][dx+1];
                    sy += g * Gy[dy+1][dx+1];
                }
                float mag = std::sqrt(sx*sx + sy*sy);
            edges[py*WIDTH + px] = mag;
            if (mag > maxEdge) maxEdge = mag;  // safe: cláusula reduction
        }
        if (omp_get_thread_num() == 0 && (py % 100 == 0 || py == HEIGHT-2))
            progress_bar("Filtro Sobel       ", py, HEIGHT-2, t.elapsed_s());
    }
    progress_bar("Filtro Sobel       ", HEIGHT-2, HEIGHT-2, t.elapsed_s());

    // c) Mezcla final (maxEdge ya está combinado → solo lectura)
    Image result(N);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        float e     = (maxEdge > 0.0f) ? edges[i] / maxEdge : 0.0f;
        float alpha = std::min(1.0f, e * 3.0f);
        result[i] = {
            (uint8_t)std::clamp((int)(blurred[i].r*(1.0f-alpha)             ), 0, 255),
            (uint8_t)std::clamp((int)(blurred[i].g*(1.0f-alpha) + 220*e*alpha), 0, 255),
            (uint8_t)std::clamp((int)(blurred[i].b*(1.0f-alpha) + 255*e*alpha), 0, 255)
        };
    }
    return result;
}

// ─────────────────────────────────────────────
//  Guardar PPM (I/O siempre secuencial)
// ─────────────────────────────────────────────
bool save_ppm(const std::string& path, const Image& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Error: no se pudo crear " << path << '\n'; return false; }
    f << "P6\n" << WIDTH << ' ' << HEIGHT << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data()), img.size() * 3);
    return f.good();
}

// ─────────────────────────────────────────────
//  Conteo de pixeles
// ─────────────────────────────────────────────
void print_stats(const Image& img, const char* label) {
    int nthreads = omp_get_max_threads();
    std::vector<std::unordered_map<uint32_t, int>> local_maps(nthreads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)img.size(); ++i) {
        uint32_t key = ((uint32_t)img[i].r << 16) |
        ((uint32_t)img[i].g <<  8) |
        (uint32_t)img[i].b;

        // Cada hilo escribe en su propio mapa — sin contención
        local_maps[omp_get_thread_num()][key]++;
    }

    // Fusionar todos los mapas locales en uno global (secuencial)
    std::unordered_map<uint32_t, int> color_count;
    for (auto& m : local_maps)
        for (auto& [key, count] : m)
            color_count[key] += count;

    std::cout << label
    << "  colores únicos: " << color_count.size() << '\n';
}

void print_stats_critical(const Image& img, const char* label) {
    std::unordered_map<uint32_t, int> color_count;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)img.size(); ++i) {
        // Empaquetar RGB en un solo entero: 0x00RRGGBB
        uint32_t key = ((uint32_t)img[i].r << 16) |
        ((uint32_t)img[i].g <<  8) |
        (uint32_t)img[i].b;

        #pragma omp critical
        color_count[key]++;
    }

    std::cout << label
    << "  colores únicos: " << color_count.size() << '\n';
}

// ─────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────
int main() {

    int chunk = 4;
    omp_sched_t schedule = omp_sched_dynamic;

    int nthreads = omp_get_max_threads();
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "  Mandelbrot 8K + Convolución 2D  — OpenMP C++\n";
    std::cout << "  Resolución      : " << WIDTH << " × " << HEIGHT << " px\n";
    std::cout << "  Iteraciones máx : " << MAX_ITER << "\n";
    std::cout << "  Hilos OpenMP    : " << nthreads << "\n";
    std::cout << "  Procesadores    : " << omp_get_num_procs() << "\n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";
    Image original;

    Timer total; total.start();
    omp_set_num_threads(16);
    // ── PASO 1 ──────────────────────────────────
    std::cout << "[ PASO 1 ] Generando fractal Mandelbrot...\n";
    Timer tp; tp.start();
    original = generate_mandelbrot(schedule, chunk);
    std::cout << "  → Tiempo: " << tp.elapsed_s() << "s\n";
    std::cout << "[ PASO 1.1 ] Conteo de pixeles - local...\n";
    tp.start();
    print_stats(original, "  Original");
    std::cout << "  → Tiempo: " << tp.elapsed_s() << "s\n";
    std::cout << "[ PASO 1.2 ] Conteo de pixeles - critical...\n";
    tp.start();
    print_stats_critical(original, "  Original");
    std::cout << "  → Tiempo: " << tp.elapsed_s() << "s\n";
    std::cout << std::endl;

    // ── PASO 2 ──────────────────────────────────
    std::cout << "\n[ PASO 2 ] Guardando mandelbrot_8k.ppm...\n";
    {
        Timer tw; tw.start();
        if (save_ppm("mandelbrot_8k.ppm", original))
            std::cout << "  → Guardado en " << tw.elapsed_s() << "s  ("
            << (WIDTH*HEIGHT*3)/1048576 << " MB)\n";
    }

    // ── PASO 3 ──────────────────────────────────
    std::cout << "\n[ PASO 3 ] Desenfoque Gaussiano (radio=15, sigma=8)...\n";
    const int   GAUSS_RADIUS = 15;
    const float GAUSS_SIGMA  = 8.0f;
    Timer tg; tg.start();
    Image blurred = gaussian_blur_separable(original, GAUSS_RADIUS, GAUSS_SIGMA);
    std::cout << "  → Tiempo Gaussiano: " << tg.elapsed_s() << "s\n";

    // ── PASO 4 ──────────────────────────────────
    std::cout << "\n[ PASO 4 ] Filtro Sobel + realce de bordes...\n";
    Timer ts; ts.start();
    Image filtered = sobel_edge_highlight(original, blurred);
    std::cout << "  → Tiempo Sobel: " << ts.elapsed_s() << "s\n";
    print_stats(filtered, "  Filtrada");

    // ── PASO 5 ──────────────────────────────────
    std::cout << "\n[ PASO 5 ] Guardando mandelbrot_8k_filtro.ppm...\n";
    {
        Timer tw; tw.start();
        if (save_ppm("mandelbrot_8k_filtro.ppm", filtered))
            std::cout << "  → Guardado en " << tw.elapsed_s() << "s\n";
    }

    double total_s = total.elapsed_s();
    std::cout << "\n═══════════════════════════════════════════════════════\n";
    std::cout << "  TIEMPO TOTAL : " << std::fixed << std::setprecision(2)
    << total_s << " segundos  (" << nthreads << " hilos)\n";
    std::cout << "  Speedup teórico máximo ~" << nthreads << "x  (Ley de Amdahl)\n";
    std::cout << "  Archivos generados:\n";
    std::cout << "    mandelbrot_8k.ppm         (original)\n";
    std::cout << "    mandelbrot_8k_filtro.ppm  (Gaussiano + Sobel)\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    return 0;
}
