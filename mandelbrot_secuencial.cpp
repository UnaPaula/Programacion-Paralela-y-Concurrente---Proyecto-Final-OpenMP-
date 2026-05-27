/*
 * ============================================================
 *  Mandelbrot 8K + Filtro de Convolución 2D (Sobel + Gaussiano)
 *  Programa puramente secuencial en C++
 * ============================================================
 *
 *  Compilar:
 *    g++ -O2 -std=c++17 -o mandelbrot mandelbrot_fractal.cpp
 *
 *  Genera dos archivos PPM (se pueden abrir con cualquier
 *  visualizador de imágenes o convertir con ImageMagick):
 *    mandelbrot_8k.ppm          — fractal original
 *    mandelbrot_8k_filtro.ppm   — con filtro aplicado
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>

// ─────────────────────────────────────────────
//  Dimensiones (8K = 7680×4320)
// ─────────────────────────────────────────────
static constexpr int WIDTH  = 7680;
static constexpr int HEIGHT = 4320;

// ─────────────────────────────────────────────
//  Parámetros del Mandelbrot
// ─────────────────────────────────────────────
static constexpr int    MAX_ITER   = 1024;
static constexpr double CENTER_X   = -0.7435669;   // zoom profundo interesante
static constexpr double CENTER_Y   =  0.1314023;
static constexpr double ZOOM       =  0.00045;      // menor → más zoom

// ─────────────────────────────────────────────
//  Tipos auxiliares
// ─────────────────────────────────────────────
struct RGB { uint8_t r, g, b; };
using Image = std::vector<RGB>;

// ─────────────────────────────────────────────
//  Utilidad: cronómetro
// ─────────────────────────────────────────────
class Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
public:
    void start() { t0 = clock::now(); }
    double elapsed_s() const {
        return std::chrono::duration<double>(clock::now() - t0).count();
    }
};

// ─────────────────────────────────────────────
//  Barra de progreso en consola
// ─────────────────────────────────────────────
void progress_bar(const char* label, int done, int total, double elapsed) {
    static const int BAR = 40;
    double frac = (double)done / total;
    int filled = (int)(frac * BAR);
    std::cout << '\r' << label << " [";
    for (int i = 0; i < BAR; ++i)
        std::cout << (i < filled ? '#' : '-');
    double eta = (done > 0) ? elapsed / frac * (1.0 - frac) : 0.0;
    std::cout << "] " << std::fixed << std::setprecision(1)
              << frac * 100.0 << "% "
              << elapsed << "s elapsed, ETA " << eta << "s   " << std::flush;
    if (done == total) std::cout << '\n';
}

// ─────────────────────────────────────────────
//  Mapa de colores: paleta "ultravioleta-fuego"
//  usando la iteración suavizada (smooth coloring)
// ─────────────────────────────────────────────
RGB iter_to_color(double smooth) {
    // Normalizar al rango [0, 1]
    double t = std::fmod(smooth / MAX_ITER * 6.0, 1.0);

    // Gradiente cíclico de 5 paradas
    struct Stop { double pos; double r, g, b; };
    static const Stop stops[] = {
        { 0.00,  0,   0,  32 },   // negro-azul profundo
        { 0.20,  0,   7, 100 },   // azul oscuro
        { 0.40, 32, 107, 203 },   // azul eléctrico
        { 0.55, 237,255,255 },    // cyan blanco
        { 0.70, 255,170,  0 },    // naranja
        { 0.85, 160,  0,  0 },    // rojo oscuro
        { 1.00,  0,   0,  32 },   // negro-azul profundo
    };
    static const int N = sizeof(stops) / sizeof(stops[0]);

    int lo = 0;
    for (int i = 0; i < N - 1; ++i)
        if (t >= stops[i].pos && t < stops[i+1].pos) { lo = i; break; }

    double range = stops[lo+1].pos - stops[lo].pos;
    double u = (range > 0) ? (t - stops[lo].pos) / range : 0.0;

    auto lerp = [&](double a, double b) { return a + u * (b - a); };
    return {
        (uint8_t)std::clamp(lerp(stops[lo].r, stops[lo+1].r), 0.0, 255.0),
        (uint8_t)std::clamp(lerp(stops[lo].g, stops[lo+1].g), 0.0, 255.0),
        (uint8_t)std::clamp(lerp(stops[lo].b, stops[lo+1].b), 0.0, 255.0)
    };
}

// ─────────────────────────────────────────────
//  Genera la imagen del conjunto de Mandelbrot
//  con suavizado de contorno (smooth iteration count)
// ─────────────────────────────────────────────
Image generate_mandelbrot() {
    Image img(WIDTH * HEIGHT);
    Timer t; t.start();

    double asp = (double)WIDTH / HEIGHT;

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
                // Smooth coloring
                double log_zn = std::log(zx*zx + zy*zy) / 2.0;
                double nu     = std::log(log_zn / std::log(2.0)) / std::log(2.0);
                double smooth = iter + 1 - nu;
                img[py * WIDTH + px] = iter_to_color(smooth);
            } else {
                img[py * WIDTH + px] = {0, 0, 0};  // interior → negro
            }
        }

        // Progreso cada 50 filas
        if (py % 50 == 0 || py == HEIGHT - 1)
            progress_bar("Generando Mandelbrot", py + 1, HEIGHT, t.elapsed_s());
    }
    return img;
}

// ─────────────────────────────────────────────
//  Construye un kernel Gaussiano normalizado
// ─────────────────────────────────────────────
std::vector<float> gaussian_kernel(int radius, float sigma) {
    int size = 2 * radius + 1;
    std::vector<float> k(size * size);
    float sum = 0;
    for (int y = -radius; y <= radius; ++y)
    for (int x = -radius; x <= radius; ++x) {
        float v = std::exp(-(x*x + y*y) / (2.0f * sigma * sigma));
        k[(y+radius)*size + (x+radius)] = v;
        sum += v;
    }
    for (auto& v : k) v /= sum;
    return k;
}

// ─────────────────────────────────────────────
//  Convolución separable 1D (horizontal + vertical)
//  para el Gaussiano — O(W·H·radius) en lugar de
//  O(W·H·radius²) que daría la versión 2D ingenua.
// ─────────────────────────────────────────────
Image gaussian_blur_separable(const Image& src, int radius, float sigma) {
    int size   = 2 * radius + 1;

    // Kernel 1D
    std::vector<float> k1d(size);
    float sum = 0;
    for (int i = -radius; i <= radius; ++i) {
        k1d[i + radius] = std::exp(-(float)(i*i) / (2.0f * sigma * sigma));
        sum += k1d[i + radius];
    }
    for (auto& v : k1d) v /= sum;

    // Buffers de trabajo en float para evitar saturación
    int N = WIDTH * HEIGHT;
    std::vector<float> tmpR(N), tmpG(N), tmpB(N);
    std::vector<float> outR(N), outG(N), outB(N);

    Timer t; t.start();

    // ── Pasada horizontal ──
    for (int py = 0; py < HEIGHT; ++py) {
        for (int px = 0; px < WIDTH; ++px) {
            float r = 0, g = 0, b = 0;
            for (int k = -radius; k <= radius; ++k) {
                int sx = std::clamp(px + k, 0, WIDTH - 1);
                const RGB& pix = src[py * WIDTH + sx];
                float w = k1d[k + radius];
                r += pix.r * w;
                g += pix.g * w;
                b += pix.b * w;
            }
            int idx = py * WIDTH + px;
            tmpR[idx] = r; tmpG[idx] = g; tmpB[idx] = b;
        }
        if (py % 100 == 0 || py == HEIGHT - 1)
            progress_bar("Gauss horizontal   ", py + 1, HEIGHT, t.elapsed_s());
    }

    // ── Pasada vertical ──
    for (int py = 0; py < HEIGHT; ++py) {
        for (int px = 0; px < WIDTH; ++px) {
            float r = 0, g = 0, b = 0;
            for (int k = -radius; k <= radius; ++k) {
                int sy = std::clamp(py + k, 0, HEIGHT - 1);
                int idx = sy * WIDTH + px;
                float w = k1d[k + radius];
                r += tmpR[idx] * w;
                g += tmpG[idx] * w;
                b += tmpB[idx] * w;
            }
            int oi = py * WIDTH + px;
            outR[oi] = r; outG[oi] = g; outB[oi] = b;
        }
        if (py % 100 == 0 || py == HEIGHT - 1)
            progress_bar("Gauss vertical     ", py + 1, HEIGHT, t.elapsed_s());
    }

    Image result(N);
    for (int i = 0; i < N; ++i)
        result[i] = {
            (uint8_t)std::clamp((int)outR[i], 0, 255),
            (uint8_t)std::clamp((int)outG[i], 0, 255),
            (uint8_t)std::clamp((int)outB[i], 0, 255)
        };
    return result;
}

// ─────────────────────────────────────────────
//  Filtro Sobel (detección de bordes)
//  combinado con la imagen original para resaltado
// ─────────────────────────────────────────────
Image sobel_edge_highlight(const Image& src, const Image& blurred) {
    //  Kernels Sobel 3×3
    static const int Gx[3][3] = {{ -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 }};
    static const int Gy[3][3] = {{ -1,-2,-1 }, {  0, 0, 0 }, {  1, 2, 1 }};

    int N = WIDTH * HEIGHT;
    // Convertir a escala de grises para el Sobel
    std::vector<float> gray(N);
    for (int i = 0; i < N; ++i)
        gray[i] = 0.2126f*src[i].r + 0.7152f*src[i].g + 0.0722f*src[i].b;

    std::vector<float> edges(N, 0);
    float maxEdge = 0;

    Timer t; t.start();

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
            if (mag > maxEdge) maxEdge = mag;
        }
        if (py % 100 == 0 || py == HEIGHT - 2)
            progress_bar("Filtro Sobel       ", py, HEIGHT - 2, t.elapsed_s());
    }

    // Mezclar: desenfoque gaussiano + bordes Sobel en cyan brillante
    Image result(N);
    for (int i = 0; i < N; ++i) {
        float e = (maxEdge > 0) ? edges[i] / maxEdge : 0.0f;  // [0,1]
        // Color del borde: azul-cyan intenso
        float eR =   0 * e, eG = 220 * e, eB = 255 * e;
        float alpha = std::min(1.0f, e * 3.0f);  // amplificar bordes finos
        result[i] = {
            (uint8_t)std::clamp((int)(blurred[i].r*(1-alpha) + eR*alpha), 0, 255),
            (uint8_t)std::clamp((int)(blurred[i].g*(1-alpha) + eG*alpha), 0, 255),
            (uint8_t)std::clamp((int)(blurred[i].b*(1-alpha) + eB*alpha), 0, 255)
        };
    }
    return result;
}

// ─────────────────────────────────────────────
//  Guardar en formato PPM (P6 binario)
// ─────────────────────────────────────────────
bool save_ppm(const std::string& path, const Image& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Error: no se pudo crear " << path << '\n'; return false; }
    f << "P6\n" << WIDTH << ' ' << HEIGHT << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data()), img.size() * 3);
    return f.good();
}

// ─────────────────────────────────────────────
//  Imprimir estadísticas de imagen
// ─────────────────────────────────────────────
void print_stats(const Image& img, const char* label) {
    long long sr = 0, sg = 0, sb = 0;
    for (auto& p : img) { sr += p.r; sg += p.g; sb += p.b; }
    double n = img.size();
    std::cout << label << "  avg R=" << sr/n
              << "  G=" << sg/n << "  B=" << sb/n << '\n';
}

// ─────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────
int main() {
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << "  Mandelbrot 8K + Convolución 2D  (secuencial C++)\n";
    std::cout << "  Resolución: " << WIDTH << " × " << HEIGHT << " px\n";
    std::cout << "  Iteraciones máximas: " << MAX_ITER << "\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";

    Timer total; total.start();

    // 1. Generar Mandelbrot
    std::cout << "[ PASO 1 ] Generando fractal Mandelbrot...\n";
    Image original = generate_mandelbrot();
    std::cout << "  → Tiempo: " << total.elapsed_s() << "s\n\n";
    print_stats(original, "  Original");

    // 2. Guardar imagen original
    std::cout << "\n[ PASO 2 ] Guardando mandelbrot_8k.ppm...\n";
    {
        Timer ts; ts.start();
        if (save_ppm("mandelbrot_8k.ppm", original))
            std::cout << "  → Guardado en " << ts.elapsed_s() << "s ("
                      << (WIDTH*HEIGHT*3)/1048576 << " MB aprox.)\n";
    }

    // 3. Desenfoque Gaussiano pesado (radio 15, sigma 8)
    std::cout << "\n[ PASO 3 ] Aplicando desenfoque Gaussiano (radio=15, sigma=8)...\n";
    const int   GAUSS_RADIUS = 15;
    const float GAUSS_SIGMA  = 8.0f;
    Timer tg; tg.start();
    Image blurred = gaussian_blur_separable(original, GAUSS_RADIUS, GAUSS_SIGMA);
    std::cout << "  → Tiempo Gaussiano: " << tg.elapsed_s() << "s\n";

    // 4. Filtro Sobel con realce de bordes sobre el desenfocado
    std::cout << "\n[ PASO 4 ] Aplicando filtro Sobel + realce de bordes...\n";
    Timer ts; ts.start();
    Image filtered = sobel_edge_highlight(original, blurred);
    std::cout << "  → Tiempo Sobel: " << ts.elapsed_s() << "s\n";
    print_stats(filtered, "  Filtrada");

    // 5. Guardar imagen filtrada
    std::cout << "\n[ PASO 5 ] Guardando mandelbrot_8k_filtro.ppm...\n";
    {
        Timer tw; tw.start();
        if (save_ppm("mandelbrot_8k_filtro.ppm", filtered))
            std::cout << "  → Guardado en " << tw.elapsed_s() << "s\n";
    }

    std::cout << "\n═══════════════════════════════════════════════════\n";
    std::cout << "  TIEMPO TOTAL: " << total.elapsed_s() << " segundos\n";
    std::cout << "  Archivos generados:\n";
    std::cout << "    mandelbrot_8k.ppm         (original)\n";
    std::cout << "    mandelbrot_8k_filtro.ppm  (Gaussiano + Sobel)\n";
    std::cout << "═══════════════════════════════════════════════════\n";

    return 0;
}
