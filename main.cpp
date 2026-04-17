#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <liquid/liquid.h>
#include <rtl-sdr.h>

// === Константы ===
constexpr size_t FFT_SIZE = 8192;
constexpr uint32_t SAMPLE_RATE = 2400000;
constexpr int INITIAL_GAIN = 439;
constexpr int WINDOW_WIDTH = 1024;
constexpr int WINDOW_HEIGHT = 768;
constexpr int SPECTRUM_HEIGHT = 200;
constexpr int WATERFALL_HEIGHT = 350;
constexpr int SCALE_HEIGHT = 50;
constexpr int PIP_WIDTH = 200;
constexpr int PIP_HEIGHT = 100;
constexpr float MIN_DB = -65.0f;
constexpr float MAX_DB = -15.0f;
constexpr double OVERLAP = 0.25;
constexpr double SCAN_START_FREQ = 0.0;
constexpr double SCAN_END_FREQ = 1'800'000'000.0;
constexpr double MIN_SPAN = 1e6;
constexpr double SCAN_STEP = SAMPLE_RATE * (1.0 - OVERLAP * 2.0);
constexpr size_t RING_BUFFER_SIZE = 16 * 1024 * 1024;
constexpr size_t TEXT_CACHE_MAX = 512;
constexpr float PEAK_DECAY_DB = 0.05f;
constexpr Uint32 FRAME_MS = 16;
constexpr int MAX_FFT_PER_FRAME = 8;

// Очистка спектра
constexpr float EDGE_REJECT_FRAC = 0.10f;    // отбрасываем по 10% с краёв FFT (roll-off)
constexpr float SMOOTH_ALPHA = 0.4f;         // коэф. IIR между циклами (0=медленно, 1=нет сглаживания)

// Детекция сигналов (CFAR)
constexpr int CFAR_WIN_HALF = 30;            // полуширина опорного окна, px
constexpr int CFAR_GUARD_HALF = 3;           // полуширина guard cells, px
constexpr float CFAR_THRESHOLD_DB = 8.0f;    // порог над средним опорного окна
constexpr int CFAR_MIN_CLUSTER_PX = 2;       // мин. ширина сигнала в px
constexpr int CFAR_MERGE_GAP_PX = 3;         // схлопывать кластеры ближе этого
constexpr size_t MAX_SIGNALS = 30;
constexpr Uint32 DETECT_INTERVAL_MS = 300;

// === Band Plan ===
struct Band {
  std::string name;
  double start_hz;
  double end_hz;
};

std::vector<Band> bands = {
    {"LW", 100000.0, 300000.0},
    {"MW", 300000.0, 3000000.0},
    {"HF (3-30)", 3000000.0, 30000000.0},
    {"CB", 27000000.0, 27400000.0},
    {"11m", 25600000.0, 26100000.0},
    {"10m", 28000000.0, 29700000.0},
    {"FM Bcast", 88000000.0, 108000000.0},
    {"2m", 144000000.0, 148000000.0},
    {"148-176", 148'000'000.0, 176'000'000.0},
    {"SAT", 230000000.0, 270000000.0},
    {"400-470", 400000000.0, 470000000.0},
    {"23cm", 1240000000.0, 1300000000.0},
};

struct DetectedSignal {
  double freq_hz;
  float peak_db;
  float snr_db;
  double width_hz;
};

// === RAII ===
struct FFTPlanWrapper {
  fftplan plan = nullptr;
  FFTPlanWrapper(size_t size, std::vector<liquid_float_complex> &in,
                 std::vector<liquid_float_complex> &out) {
    plan = fft_create_plan(size, in.data(), out.data(), LIQUID_FFT_FORWARD, 0);
  }
  ~FFTPlanWrapper() { if (plan) fft_destroy_plan(plan); }
};

struct RTLSDRDevice {
  rtlsdr_dev_t *dev = nullptr;
  ~RTLSDRDevice() { if (dev) rtlsdr_close(dev); }
};

// === Кольцевой буфер (SPSC) ===
class RingBuffer {
  std::vector<uint8_t> buffer;
  std::atomic<size_t> read_pos{0};
  std::atomic<size_t> write_pos{0};
  size_t size;

public:
  RingBuffer(size_t s) : buffer(s), size(s) {}

  size_t available_read() const {
    size_t wp = write_pos.load(std::memory_order_acquire);
    size_t rp = read_pos.load(std::memory_order_acquire);
    return wp >= rp ? wp - rp : wp + size - rp;
  }

  size_t available_write() const { return size - available_read() - 1; }

  size_t write(const uint8_t *data, size_t len) {
    size_t avail = available_write();
    if (len > avail) len = avail;
    size_t wp = write_pos.load(std::memory_order_relaxed);
    size_t to_end = size - wp;
    if (len <= to_end) {
      std::memcpy(&buffer[wp], data, len);
    } else {
      std::memcpy(&buffer[wp], data, to_end);
      std::memcpy(&buffer[0], data + to_end, len - to_end);
    }
    write_pos.store((wp + len) % size, std::memory_order_release);
    return len;
  }

  size_t read(uint8_t *data, size_t len) {
    size_t avail = available_read();
    if (len > avail) len = avail;
    size_t rp = read_pos.load(std::memory_order_relaxed);
    size_t to_end = size - rp;
    if (len <= to_end) {
      std::memcpy(data, &buffer[rp], len);
    } else {
      std::memcpy(data, &buffer[rp], to_end);
      std::memcpy(data + to_end, &buffer[0], len - to_end);
    }
    read_pos.store((rp + len) % size, std::memory_order_release);
    return len;
  }
};

// === Кэш текстур текста ===
class TextCache {
  struct Entry { SDL_Texture *tex; int w, h; };
  std::unordered_map<std::string, Entry> cache;
  TTF_Font *font;
  SDL_Renderer *renderer;

public:
  TextCache(TTF_Font *f, SDL_Renderer *r) : font(f), renderer(r) {}
  ~TextCache() { clear(); }

  void clear() {
    for (auto &kv : cache) SDL_DestroyTexture(kv.second.tex);
    cache.clear();
  }

  const Entry *entry(const char *text, SDL_Color c) {
    if (!font || !text) return nullptr;
    char key[256];
    std::snprintf(key, sizeof(key), "%u,%u,%u|%s", c.r, c.g, c.b, text);
    auto it = cache.find(key);
    if (it != cache.end()) return &it->second;
    if (cache.size() >= TEXT_CACHE_MAX) clear();
    SDL_Surface *surf = TTF_RenderText_Solid(font, text, c);
    if (!surf) return nullptr;
    SDL_Texture *tex = SDL_CreateTextureFromSurface(renderer, surf);
    int tw = surf->w, th = surf->h;
    SDL_FreeSurface(surf);
    if (!tex) return nullptr;
    return &cache.emplace(key, Entry{tex, tw, th}).first->second;
  }

  void render(const char *t, SDL_Color c, int x, int y) {
    auto e = entry(t, c);
    if (!e) return;
    SDL_Rect d{x, y, e->w, e->h};
    SDL_RenderCopy(renderer, e->tex, nullptr, &d);
  }

  bool measure(const char *t, SDL_Color c, int &w, int &h) {
    auto e = entry(t, c);
    if (!e) return false;
    w = e->w; h = e->h;
    return true;
  }
};

// === Глобалки для worker↔main ===
std::atomic<bool> worker_running{true};
RingBuffer sample_buffer(RING_BUFFER_SIZE);

std::atomic<double> scan_start_freq{SCAN_START_FREQ};
std::atomic<double> scan_end_freq{SCAN_END_FREQ};
double display_start_freq = SCAN_START_FREQ;
double display_end_freq = SCAN_END_FREQ;

std::atomic<double> current_center_freq{SCAN_START_FREQ};
std::atomic<bool> pip_mode{false};
std::atomic<double> pip_center_freq{0.0};
std::atomic<bool> full_main_complete{false};
std::atomic<bool> this_step_is_pip{false};

std::atomic<int> sdr_gain{INITIAL_GAIN};
std::atomic<bool> gain_changed{false};

// === Утилиты ===
float u8_to_float(uint8_t v) { return ((float)v - 127.5f) / 128.0f; }

std::vector<float> make_hann_window(size_t N) {
  std::vector<float> w(N);
  for (size_t i = 0; i < N; ++i)
    w[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * (float)i / (float)(N - 1)));
  return w;
}

float lin_to_db(float x) { return 10.0f * std::log10(std::max(x, 1e-30f)); }

void classic_gradient(float t, Uint8 &r, Uint8 &g, Uint8 &b) {
  t = std::clamp(t, 0.0f, 1.0f);
  if (t < 0.2f) {
    float v = t / 0.2f; r = 0; g = 0; b = (Uint8)(v * 255);
  } else if (t < 0.4f) {
    float v = (t - 0.2f) / 0.2f; r = 0; g = (Uint8)(v * 255); b = 255;
  } else if (t < 0.6f) {
    float v = (t - 0.4f) / 0.2f; r = 0; g = 255; b = (Uint8)((1.0f - v) * 255);
  } else if (t < 0.8f) {
    float v = (t - 0.6f) / 0.2f; r = (Uint8)(v * 255); g = 255; b = 0;
  } else {
    float v = (t - 0.8f) / 0.2f;
    if (v < 0.5f) {
      r = 255; g = (Uint8)((1.0f - v * 2.0f) * 255); b = 0;
    } else {
      float w = (v - 0.5f) * 2.0f;
      r = 255; g = (Uint8)(w * 255); b = (Uint8)(w * 255);
    }
  }
}

void clamp_display_range() {
  display_start_freq = std::max(SCAN_START_FREQ, display_start_freq);
  display_end_freq = std::min(SCAN_END_FREQ, display_end_freq);
  if (display_end_freq - display_start_freq < MIN_SPAN) {
    double c = (display_start_freq + display_end_freq) / 2.0;
    display_start_freq = std::max(SCAN_START_FREQ, c - MIN_SPAN / 2);
    display_end_freq = std::min(SCAN_END_FREQ, c + MIN_SPAN / 2);
  }
}

void update_scan_range_from_display() {
  double s = std::max(SCAN_START_FREQ, display_start_freq - SAMPLE_RATE * 0.6);
  double e = std::min(SCAN_END_FREQ, display_end_freq + SAMPLE_RATE * 0.6);
  constexpr double MIN_SCAN_SPAN = 10'000'000.0;
  if (e - s < MIN_SCAN_SPAN) {
    double c = (s + e) / 2.0;
    s = std::max(SCAN_START_FREQ, c - MIN_SCAN_SPAN / 2);
    e = std::min(SCAN_END_FREQ, c + MIN_SCAN_SPAN / 2);
  }
  scan_start_freq.store(s);
  scan_end_freq.store(e);
}

// Форматирование частоты с автошагом (MHz/kHz)
void format_freq(double hz, char *buf, size_t n) {
  if (hz >= 1e9)
    std::snprintf(buf, n, "%.3f GHz", hz / 1e9);
  else if (hz >= 1e6)
    std::snprintf(buf, n, "%.3f MHz", hz / 1e6);
  else
    std::snprintf(buf, n, "%.1f kHz", hz / 1e3);
}

void print_help() {
  std::cout
      << "=== Wide-band Spectrum Analyzer ===\n"
      << "Right drag  : panning\n"
      << "Wheel       : zoom (+Shift = fine)\n"
      << "Left click  : quick-view 1 MHz (+Shift = 100 kHz)\n"
      << "Left (scale): выбор бэнда\n"
      << "Arrows/Home : навигация\n"
      << "P           : PiP в позиции курсора\n"
      << "H / C       : peak hold / clear peak\n"
      << "M           : межцикловое IIR сглаживание\n"
      << "N           : линия шумового пола\n"
      << "D           : детекция сигналов (CFAR)\n"
      << "+ / -       : gain\n"
      << "1..9        : быстрый выбор бэнда\n"
      << "F1 / ?      : справка\n"
      << "Q / Esc     : выход\n\n";
}

// === Worker thread ===
void sdr_worker() {
  RTLSDRDevice sdr;
  if (rtlsdr_open(&sdr.dev, 0) < 0) {
    std::cerr << "Failed to open RTL-SDR\n";
    worker_running = false;
    return;
  }
  if (rtlsdr_set_sample_rate(sdr.dev, SAMPLE_RATE) < 0)
    std::cerr << "warn: set_sample_rate\n";
  if (rtlsdr_set_tuner_gain_mode(sdr.dev, 1) < 0)
    std::cerr << "warn: set_tuner_gain_mode\n";
  if (rtlsdr_set_tuner_gain(sdr.dev, sdr_gain.load()) < 0)
    std::cerr << "warn: set_tuner_gain\n";
  rtlsdr_set_agc_mode(sdr.dev, 0);

  double main_center = scan_start_freq.load();
  rtlsdr_set_center_freq(sdr.dev, (uint32_t)main_center);
  rtlsdr_reset_buffer(sdr.dev);
  usleep(10000);

  bool alternate = false;
  std::vector<uint8_t> block(FFT_SIZE * 2);

  while (worker_running.load()) {
    if (gain_changed.exchange(false))
      rtlsdr_set_tuner_gain(sdr.dev, sdr_gain.load());

    double s = scan_start_freq.load();
    double e = scan_end_freq.load();
    double next_center;
    bool do_pip = false;

    if (pip_mode.load() && (alternate = !alternate)) {
      next_center = pip_center_freq.load();
      do_pip = true;
    } else {
      next_center = main_center + SCAN_STEP;
      if (next_center > e || next_center < s) {
        next_center = s;
        full_main_complete.store(true);
      }
      main_center = next_center;
    }

    rtlsdr_set_center_freq(sdr.dev, (uint32_t)next_center);
    current_center_freq.store(next_center);
    this_step_is_pip.store(do_pip);
    usleep(1000);

    int total = 0;
    while (total < (int)block.size() && worker_running.load()) {
      int n = 0;
      int ret = rtlsdr_read_sync(sdr.dev, block.data() + total,
                                 (int)block.size() - total, &n);
      if (ret < 0 || n <= 0) { usleep(1000); continue; }
      total += n;
    }
    if (!worker_running.load()) break;

    if (sample_buffer.write(block.data(), block.size()) < block.size())
      usleep(1000);
  }
}

int main() {
  print_help();

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cerr << "SDL_Init: " << SDL_GetError() << "\n";
    return 1;
  }
  if (TTF_Init() < 0) {
    std::cerr << "TTF_Init: " << TTF_GetError() << "\n";
    SDL_Quit();
    return 1;
  }

  auto window_func = make_hann_window(FFT_SIZE);
  float win_sum_sq = 0.0f;
  for (float v : window_func) win_sum_sq += v * v;

  const double bin_width = (double)SAMPLE_RATE / FFT_SIZE;
  const float fft_norm = 1.0f / (win_sum_sq * (float)bin_width);

  std::vector<liquid_float_complex> fft_in(FFT_SIZE), fft_out(FFT_SIZE);
  FFTPlanWrapper fft_wrapper(FFT_SIZE, fft_in, fft_out);
  if (!fft_wrapper.plan) {
    std::cerr << "FFT plan failed\n";
    TTF_Quit(); SDL_Quit();
    return 1;
  }

  std::thread worker_thread(sdr_worker);

  size_t total_bins =
      (size_t)std::ceil((SCAN_END_FREQ - SCAN_START_FREQ) / bin_width);
  std::vector<float> cycle_psd_sum(total_bins, 0.0f);
  std::vector<int> cycle_count(total_bins, 0);
  std::vector<float> display_scan(total_bins, MIN_DB - 100.0f);
  std::vector<float> display_peak(total_bins, MIN_DB - 100.0f);
  bool peak_hold = false;
  bool smoothing_enabled = true;
  bool show_noise_floor = false;
  bool detection_enabled = false;

  std::vector<DetectedSignal> signals;
  float noise_floor_db = MIN_DB;
  Uint32 last_detect_ms = 0;

  SDL_Window *win = SDL_CreateWindow(
      "Spectrum Analyzer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
      WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
  SDL_Renderer *renderer = win ? SDL_CreateRenderer(win, -1,
      SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC) : nullptr;
  if (!win || !renderer) {
    std::cerr << "SDL window/renderer failed\n";
    worker_running = false;
    worker_thread.join();
    if (renderer) SDL_DestroyRenderer(renderer);
    if (win) SDL_DestroyWindow(win);
    TTF_Quit(); SDL_Quit();
    return 1;
  }
  SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

  TTF_Font *font =
      TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12);
  if (!font) font = TTF_OpenFont("/System/Library/Fonts/SFNS.ttf", 12);
  if (!font) font = TTF_OpenFont("C:\\Windows\\Fonts\\arial.ttf", 12);
  if (!font) std::cerr << "warn: font not found\n";

  TextCache text_cache(font, renderer);

  SDL_Texture *waterfall_tex = SDL_CreateTexture(
      renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
      WINDOW_WIDTH, WATERFALL_HEIGHT);
  SDL_Texture *pip_tex = SDL_CreateTexture(
      renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
      PIP_WIDTH, PIP_HEIGHT);

  std::vector<std::vector<Uint32>> waterfall_lines(
      WATERFALL_HEIGHT, std::vector<Uint32>(WINDOW_WIDTH, 0));
  std::vector<std::vector<Uint32>> pip_waterfall_lines(
      PIP_HEIGHT, std::vector<Uint32>(PIP_WIDTH, 0));
  int waterfall_top = 0;
  int pip_top = 0;

  bool quit = false;
  int mouse_x = -1, mouse_y = -1;
  bool dragging = false;
  int drag_x0 = 0;
  double drag_start0 = 0, drag_end0 = 0;

  const double MAX_SPAN = SCAN_END_FREQ - SCAN_START_FREQ;
  std::vector<uint8_t> rbuf(FFT_SIZE * 2);
  std::vector<float> samp_I(FFT_SIZE), samp_Q(FFT_SIZE);
  std::vector<float> px_db(WINDOW_WIDTH, MIN_DB - 100.0f);

  update_scan_range_from_display();

  double prev_scan_s = SCAN_START_FREQ, prev_scan_e = SCAN_END_FREQ;
  bool prev_pip = false;

  const SDL_Color COL_WHITE{255, 255, 255, 255};
  const SDL_Color COL_YELLOW{255, 230, 80, 255};
  const SDL_Color COL_CYAN{120, 200, 220, 255};
  const SDL_Color COL_ORANGE{255, 160, 40, 255};
  const SDL_Color COL_GREEN{120, 230, 140, 255};
  const SDL_Color COL_SCALE_TEXT{200, 210, 220, 255};

  // === Обработка одного FFT-блока с очисткой спектра ===
  auto process_fft = [&]() {
    sample_buffer.read(rbuf.data(), rbuf.size());
    double cf = current_center_freq.load();
    double cs = scan_start_freq.load();
    double ce = scan_end_freq.load();
    bool pm = pip_mode.load();

    if (pm != prev_pip || cs != prev_scan_s || ce != prev_scan_e) {
      std::fill(cycle_psd_sum.begin(), cycle_psd_sum.end(), 0.0f);
      std::fill(cycle_count.begin(), cycle_count.end(), 0);
      prev_pip = pm;
      prev_scan_s = cs;
      prev_scan_e = ce;
    }

    // DC removal: вычитаем средние I и Q перед окном
    float sum_I = 0, sum_Q = 0;
    for (size_t i = 0; i < FFT_SIZE; ++i) {
      samp_I[i] = u8_to_float(rbuf[2 * i]);
      samp_Q[i] = u8_to_float(rbuf[2 * i + 1]);
      sum_I += samp_I[i];
      sum_Q += samp_Q[i];
    }
    float dc_I = sum_I / FFT_SIZE;
    float dc_Q = sum_Q / FFT_SIZE;
    for (size_t i = 0; i < FFT_SIZE; ++i) {
      float I = (samp_I[i] - dc_I) * window_func[i];
      float Q = (samp_Q[i] - dc_Q) * window_func[i];
      fft_in[i] = liquid_float_complex{I, Q};
    }
    fft_execute(fft_wrapper.plan);
    fft_shift(fft_out.data(), FFT_SIZE);

    // DC spike killer: интерполяция центрального бина
    size_t c = FFT_SIZE / 2;
    if (c > 0 && c < FFT_SIZE - 1) {
      fft_out[c].real = 0.5f * (fft_out[c - 1].real + fft_out[c + 1].real);
      fft_out[c].imag = 0.5f * (fft_out[c - 1].imag + fft_out[c + 1].imag);
    }

    if (this_step_is_pip.load()) {
      // PiP: peak-hold по бинам, попадающим в один пиксель
      std::vector<Uint32> line(PIP_WIDTH, 0);
      for (int px = 0; px < PIP_WIDTH; ++px) {
        size_t k_lo = (size_t)std::floor((double)px / PIP_WIDTH * FFT_SIZE);
        size_t k_hi =
            (size_t)std::ceil((double)(px + 1) / PIP_WIDTH * FFT_SIZE);
        if (k_hi > FFT_SIZE) k_hi = FFT_SIZE;
        if (k_lo >= k_hi) k_hi = k_lo + 1;
        float max_psd = 0.0f;
        for (size_t k = k_lo; k < k_hi && k < FFT_SIZE; ++k) {
          float re = fft_out[k].real, im = fft_out[k].imag;
          float psd = (re * re + im * im) * fft_norm;
          if (psd > max_psd) max_psd = psd;
        }
        float db = lin_to_db(max_psd);
        float n = std::clamp((db - MIN_DB) / (MAX_DB - MIN_DB), 0.0f, 1.0f);
        Uint8 r, g, b;
        classic_gradient(n, r, g, b);
        line[px] = (255U << 24) | (r << 16) | (g << 8) | b;
      }
      pip_top = (pip_top - 1 + PIP_HEIGHT) % PIP_HEIGHT;
      pip_waterfall_lines[pip_top] = std::move(line);
    } else {
      // Main: накопление + edge rejection + IIR между циклами
      const size_t edge = (size_t)(FFT_SIZE * EDGE_REJECT_FRAC);
      const float alpha = smoothing_enabled ? SMOOTH_ALPHA : 1.0f;
      for (size_t k = edge; k < FFT_SIZE - edge; ++k) {
        float re = fft_out[k].real, im = fft_out[k].imag;
        float psd = (re * re + im * im) * fft_norm;
        double bin_freq = cf + (k - FFT_SIZE / 2.0) * bin_width;
        if (bin_freq < SCAN_START_FREQ || bin_freq >= SCAN_END_FREQ) continue;
        size_t idx =
            (size_t)std::round((bin_freq - SCAN_START_FREQ) / bin_width);
        if (idx >= total_bins) continue;
        cycle_psd_sum[idx] += psd;
        cycle_count[idx]++;
        float avg_db = lin_to_db(cycle_psd_sum[idx] / cycle_count[idx]);
        if (display_scan[idx] <= MIN_DB - 99.0f)
          display_scan[idx] = avg_db;
        else
          display_scan[idx] =
              alpha * avg_db + (1.0f - alpha) * display_scan[idx];
        if (peak_hold && display_scan[idx] > display_peak[idx])
          display_peak[idx] = display_scan[idx];
      }
    }
  };

  auto on_full_scan_complete = [&]() {
    std::fill(cycle_psd_sum.begin(), cycle_psd_sum.end(), 0.0f);
    std::fill(cycle_count.begin(), cycle_count.end(), 0);
    if (peak_hold)
      for (auto &v : display_peak) v -= PEAK_DECAY_DB;

    std::vector<Uint32> line(WINDOW_WIDTH, 0);
    const float range = MAX_DB - MIN_DB;
    double span = display_end_freq - display_start_freq;
    for (int x = 0; x < WINDOW_WIDTH; ++x) {
      double f_lo = display_start_freq + (double)x / WINDOW_WIDTH * span;
      double f_hi = display_start_freq + (double)(x + 1) / WINDOW_WIDTH * span;
      size_t lo = (size_t)std::floor((f_lo - SCAN_START_FREQ) / bin_width);
      size_t hi = (size_t)std::ceil((f_hi - SCAN_START_FREQ) / bin_width);
      float db = MIN_DB - 100.0f;
      for (size_t i = lo; i < hi && i < total_bins; ++i)
        db = std::max(db, display_scan[i]);
      float n = std::clamp((db - MIN_DB) / range, 0.0f, 1.0f);
      Uint8 r, g, b;
      classic_gradient(n, r, g, b);
      line[x] = (255U << 24) | (r << 16) | (g << 8) | b;
    }
    waterfall_top = (waterfall_top - 1 + WATERFALL_HEIGHT) % WATERFALL_HEIGHT;
    waterfall_lines[waterfall_top] = std::move(line);
  };

  auto blit_waterfall = [&](SDL_Texture *tex,
                            std::vector<std::vector<Uint32>> &lines,
                            int width, int height, int top) {
    if (!tex) return;
    void *pixels;
    int pitch;
    if (SDL_LockTexture(tex, nullptr, &pixels, &pitch) != 0) return;
    for (int y = 0; y < height; ++y) {
      int src_y = (top + y) % height;
      Uint8 *row = (Uint8 *)pixels + y * pitch;
      std::memcpy(row, lines[src_y].data(), width * sizeof(Uint32));
    }
    SDL_UnlockTexture(tex);
  };

  // Шумовой пол: 20-й процентиль по px_db
  auto compute_noise_floor = [&]() {
    std::vector<float> copy(px_db);
    std::sort(copy.begin(), copy.end());
    noise_floor_db = copy[WINDOW_WIDTH / 5];
  };

  // CFAR детектор: возвращает список кластерных пиков
  auto detect_signals_cfar = [&]() {
    signals.clear();
    std::vector<bool> is_peak(WINDOW_WIDTH, false);

    for (int px = CFAR_WIN_HALF; px < WINDOW_WIDTH - CFAR_WIN_HALF; ++px) {
      float sum = 0;
      int cnt = 0;
      for (int k = px - CFAR_WIN_HALF; k < px - CFAR_GUARD_HALF; ++k) {
        sum += px_db[k]; cnt++;
      }
      for (int k = px + CFAR_GUARD_HALF + 1; k <= px + CFAR_WIN_HALF; ++k) {
        sum += px_db[k]; cnt++;
      }
      if (cnt == 0) continue;
      float mean = sum / cnt;
      if (px_db[px] > mean + CFAR_THRESHOLD_DB &&
          px_db[px] > MIN_DB + 2.0f)
        is_peak[px] = true;
    }

    // Заполнить короткие пропуски между is_peak (схлопывание близких кластеров)
    for (int px = 1; px < WINDOW_WIDTH - 1; ++px) {
      if (is_peak[px]) continue;
      int left = -1, right = -1;
      for (int k = px - 1; k >= std::max(0, px - CFAR_MERGE_GAP_PX); --k)
        if (is_peak[k]) { left = k; break; }
      for (int k = px + 1;
           k <= std::min(WINDOW_WIDTH - 1, px + CFAR_MERGE_GAP_PX); ++k)
        if (is_peak[k]) { right = k; break; }
      if (left >= 0 && right >= 0) is_peak[px] = true;
    }

    // Кластеризация в сигналы
    int start = -1;
    double span = display_end_freq - display_start_freq;
    for (int px = 0; px <= WINDOW_WIDTH; ++px) {
      bool p = (px < WINDOW_WIDTH) ? is_peak[px] : false;
      if (p && start < 0) start = px;
      else if (!p && start >= 0) {
        int end_px = px;
        int width = end_px - start;
        if (width >= CFAR_MIN_CLUSTER_PX) {
          int peak_x = start;
          float peak_val = px_db[start];
          for (int k = start + 1; k < end_px; ++k) {
            if (px_db[k] > peak_val) { peak_val = px_db[k]; peak_x = k; }
          }
          DetectedSignal s;
          s.freq_hz = display_start_freq +
                      ((double)peak_x + 0.5) / WINDOW_WIDTH * span;
          s.peak_db = peak_val;
          s.snr_db = peak_val - noise_floor_db;
          s.width_hz = (double)width / WINDOW_WIDTH * span;
          signals.push_back(s);
        }
        start = -1;
      }
    }

    std::sort(signals.begin(), signals.end(),
              [](const DetectedSignal &a, const DetectedSignal &b) {
                return a.peak_db > b.peak_db;
              });
    if (signals.size() > MAX_SIGNALS) signals.resize(MAX_SIGNALS);
  };

  Uint32 last_frame_ms = SDL_GetTicks();
  Uint32 last_waterfall_blit_ms = 0;

  while (!quit) {
    if (!worker_running.load() && sample_buffer.available_read() < rbuf.size())
      break;

    // === 1. FFT batch-ом ===
    int processed = 0;
    while (processed < MAX_FFT_PER_FRAME &&
           sample_buffer.available_read() >= rbuf.size()) {
      process_fft();
      processed++;
    }

    if (full_main_complete.load()) {
      full_main_complete.store(false);
      on_full_scan_complete();
    }

    // === 2. События ===
    double display_total_hz = display_end_freq - display_start_freq;

    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_QUIT) { quit = true; continue; }

      SDL_Keymod mods = SDL_GetModState();
      bool shift = mods & KMOD_SHIFT;

      if (e.type == SDL_MOUSEMOTION) {
        mouse_x = e.motion.x;
        mouse_y = e.motion.y;
        if (dragging) {
          double d = (mouse_x - drag_x0) / (double)WINDOW_WIDTH *
                     display_total_hz;
          display_start_freq = drag_start0 - d;
          display_end_freq = drag_end0 - d;
          clamp_display_range();
          update_scan_range_from_display();
          display_total_hz = display_end_freq - display_start_freq;
        }
      } else if (e.type == SDL_MOUSEBUTTONDOWN) {
        if (e.button.button == SDL_BUTTON_RIGHT) {
          dragging = true;
          drag_x0 = e.button.x;
          drag_start0 = display_start_freq;
          drag_end0 = display_end_freq;
        } else if (e.button.button == SDL_BUTTON_LEFT) {
          if (e.button.y > WINDOW_HEIGHT - SCALE_HEIGHT) {
            double f = display_start_freq +
                       (double)e.button.x / WINDOW_WIDTH * display_total_hz;
            for (const auto &b : bands) {
              if (f >= b.start_hz && f <= b.end_hz) {
                display_start_freq = b.start_hz;
                display_end_freq = b.end_hz;
                clamp_display_range();
                update_scan_range_from_display();
                break;
              }
            }
          } else if (e.button.y < SPECTRUM_HEIGHT + WATERFALL_HEIGHT) {
            double f = display_start_freq +
                       (double)e.button.x / WINDOW_WIDTH * display_total_hz;
            double span = shift ? 100e3 : 1e6;
            display_start_freq = f - span / 2;
            display_end_freq = f + span / 2;
            clamp_display_range();
            update_scan_range_from_display();
          }
        }
      } else if (e.type == SDL_MOUSEBUTTONUP) {
        if (e.button.button == SDL_BUTTON_RIGHT) dragging = false;
      } else if (e.type == SDL_MOUSEWHEEL) {
        double cur_f = display_start_freq +
                       (double)mouse_x / WINDOW_WIDTH * display_total_hz;
        double span = display_end_freq - display_start_freq;
        double z = shift ? 0.9 : 0.7;
        double zf = (e.wheel.y > 0) ? z : 1.0 / z;
        double new_span = std::clamp(span * zf, MIN_SPAN, MAX_SPAN);
        double ratio = (cur_f - display_start_freq) / span;
        display_start_freq = cur_f - ratio * new_span;
        display_end_freq = display_start_freq + new_span;
        if (display_start_freq < SCAN_START_FREQ) {
          display_start_freq = SCAN_START_FREQ;
          display_end_freq = SCAN_START_FREQ + new_span;
        }
        if (display_end_freq > SCAN_END_FREQ) {
          display_end_freq = SCAN_END_FREQ;
          display_start_freq = SCAN_END_FREQ - new_span;
        }
        clamp_display_range();
        update_scan_range_from_display();
      } else if (e.type == SDL_KEYDOWN) {
        SDL_Keycode sym = e.key.keysym.sym;
        double scroll = display_total_hz * 0.1;

        if (sym == SDLK_LEFT) {
          display_start_freq -= scroll;
          display_end_freq -= scroll;
          if (display_start_freq < SCAN_START_FREQ) {
            double d = SCAN_START_FREQ - display_start_freq;
            display_start_freq = SCAN_START_FREQ;
            display_end_freq += d;
          }
          clamp_display_range();
          update_scan_range_from_display();
        } else if (sym == SDLK_RIGHT) {
          display_start_freq += scroll;
          display_end_freq += scroll;
          if (display_end_freq > SCAN_END_FREQ) {
            double d = display_end_freq - SCAN_END_FREQ;
            display_end_freq = SCAN_END_FREQ;
            display_start_freq -= d;
          }
          clamp_display_range();
          update_scan_range_from_display();
        } else if (sym == SDLK_HOME) {
          display_start_freq = SCAN_START_FREQ;
          display_end_freq = SCAN_END_FREQ;
          clamp_display_range();
          update_scan_range_from_display();
        } else if (sym == SDLK_p) {
          bool nm = !pip_mode.load();
          pip_mode.store(nm);
          if (nm) {
            double mf = display_start_freq +
                        (double)mouse_x / WINDOW_WIDTH * display_total_hz;
            pip_center_freq.store(mf);
            for (auto &l : pip_waterfall_lines)
              std::fill(l.begin(), l.end(), 0U);
            pip_top = 0;
          }
        } else if (sym == SDLK_h) {
          peak_hold = !peak_hold;
          if (peak_hold)
            std::fill(display_peak.begin(), display_peak.end(),
                      MIN_DB - 100.0f);
        } else if (sym == SDLK_c) {
          std::fill(display_peak.begin(), display_peak.end(),
                    MIN_DB - 100.0f);
        } else if (sym == SDLK_m) {
          smoothing_enabled = !smoothing_enabled;
        } else if (sym == SDLK_n) {
          show_noise_floor = !show_noise_floor;
        } else if (sym == SDLK_d) {
          detection_enabled = !detection_enabled;
          if (!detection_enabled) signals.clear();
        } else if (sym == SDLK_PLUS || sym == SDLK_EQUALS ||
                   sym == SDLK_KP_PLUS) {
          sdr_gain.fetch_add(10);
          gain_changed.store(true);
        } else if (sym == SDLK_MINUS || sym == SDLK_KP_MINUS) {
          int g = sdr_gain.load();
          if (g > 0) {
            sdr_gain.store(std::max(0, g - 10));
            gain_changed.store(true);
          }
        } else if (sym == SDLK_ESCAPE || sym == SDLK_q) {
          quit = true;
        } else if (sym == SDLK_F1 || sym == SDLK_SLASH) {
          print_help();
        } else if (sym >= SDLK_1 && sym <= SDLK_9) {
          size_t idx = (size_t)(sym - SDLK_1);
          if (idx < bands.size()) {
            display_start_freq = bands[idx].start_hz;
            display_end_freq = bands[idx].end_hz;
            clamp_display_range();
            update_scan_range_from_display();
          }
        }
      }
    }

    // === 3. Рендер ===
    Uint32 now = SDL_GetTicks();
    if (now - last_frame_ms < FRAME_MS) {
      if (processed == 0) SDL_Delay(1);
      continue;
    }
    last_frame_ms = now;

    if (now - last_waterfall_blit_ms >= 30) {
      blit_waterfall(waterfall_tex, waterfall_lines, WINDOW_WIDTH,
                     WATERFALL_HEIGHT, waterfall_top);
      blit_waterfall(pip_tex, pip_waterfall_lines, PIP_WIDTH, PIP_HEIGHT,
                     pip_top);
      last_waterfall_blit_ms = now;
    }

    display_total_hz = display_end_freq - display_start_freq;
    double display_total_mhz = display_total_hz / 1e6;
    double start_mhz = display_start_freq / 1e6;
    double end_mhz = display_end_freq / 1e6;

    auto freq_to_px = [&](double f) {
      return (int)((f - display_start_freq) / display_total_hz * WINDOW_WIDTH);
    };
    auto peak_db_at = [&](const std::vector<float> &src, int x) {
      double f_lo = display_start_freq +
                    (double)x / WINDOW_WIDTH * display_total_hz;
      double f_hi = display_start_freq +
                    (double)(x + 1) / WINDOW_WIDTH * display_total_hz;
      size_t lo = (size_t)std::floor((f_lo - SCAN_START_FREQ) / bin_width);
      size_t hi = (size_t)std::ceil((f_hi - SCAN_START_FREQ) / bin_width);
      float db = MIN_DB - 100.0f;
      for (size_t i = lo; i < hi && i < total_bins; ++i)
        db = std::max(db, src[i]);
      return db;
    };

    // Собираем спектр по пикселям (используется для отрисовки + детекции)
    for (int x = 0; x < WINDOW_WIDTH; ++x)
      px_db[x] = peak_db_at(display_scan, x);

    // Периодическое обновление NF + детекции
    if (now - last_detect_ms >= DETECT_INTERVAL_MS) {
      compute_noise_floor();
      if (detection_enabled) detect_signals_cfar();
      last_detect_ms = now;
    }

    const float db_range = MAX_DB - MIN_DB;

    // --- Фон ---
    SDL_SetRenderDrawColor(renderer, 12, 16, 24, 255);
    SDL_RenderClear(renderer);

    double grid_step_mhz = 10.0;
    if (display_total_mhz > 500) grid_step_mhz = 100.0;
    else if (display_total_mhz > 100) grid_step_mhz = 50.0;
    else if (display_total_mhz < 1) grid_step_mhz = 0.1;
    else if (display_total_mhz < 10) grid_step_mhz = 1.0;

    double start_r = std::ceil(start_mhz / grid_step_mhz) * grid_step_mhz;
    double end_r = std::floor(end_mhz / grid_step_mhz) * grid_step_mhz;

    // Вертикальная сетка
    SDL_SetRenderDrawColor(renderer, 40, 50, 60, 255);
    for (double mhz = start_r; mhz <= end_r; mhz += grid_step_mhz) {
      int x = freq_to_px(mhz * 1e6);
      if (x >= 0 && x < WINDOW_WIDTH)
        SDL_RenderDrawLine(renderer, x, 0, x, SPECTRUM_HEIGHT);
    }

    // Горизонтальная сетка dB (пунктир)
    SDL_SetRenderDrawColor(renderer, 40, 50, 60, 255);
    constexpr int DB_STEPS = 4;
    for (int i = 1; i < DB_STEPS; ++i) {
      int y = i * SPECTRUM_HEIGHT / DB_STEPS;
      for (int x = 0; x < WINDOW_WIDTH; x += 4)
        SDL_RenderDrawPoint(renderer, x, y);
    }

    // Линия шумового пола
    if (show_noise_floor) {
      float nf_norm =
          std::clamp((noise_floor_db - MIN_DB) / db_range, 0.0f, 1.0f);
      int y = (int)(SPECTRUM_HEIGHT - nf_norm * SPECTRUM_HEIGHT);
      SDL_SetRenderDrawColor(renderer, 80, 200, 120, 160);
      for (int x = 0; x < WINDOW_WIDTH; x += 3)
        SDL_RenderDrawPoint(renderer, x, y);
    }

    // --- Спектр: заливка градиентом + кромка ---
    for (int x = 0; x < WINDOW_WIDTH; ++x) {
      float n = std::clamp((px_db[x] - MIN_DB) / db_range, 0.0f, 1.0f);
      int y = (int)(SPECTRUM_HEIGHT - n * SPECTRUM_HEIGHT);
      Uint8 r, g, b;
      classic_gradient(n, r, g, b);
      SDL_SetRenderDrawColor(renderer, r, g, b, 140);
      SDL_RenderDrawLine(renderer, x, y, x, SPECTRUM_HEIGHT);
    }
    SDL_SetRenderDrawColor(renderer, 230, 240, 255, 220);
    {
      int prev_y = SPECTRUM_HEIGHT;
      for (int x = 0; x < WINDOW_WIDTH; ++x) {
        float n = std::clamp((px_db[x] - MIN_DB) / db_range, 0.0f, 1.0f);
        int y = (int)(SPECTRUM_HEIGHT - n * SPECTRUM_HEIGHT);
        if (x > 0) SDL_RenderDrawLine(renderer, x - 1, prev_y, x, y);
        prev_y = y;
      }
    }

    // Peak hold
    if (peak_hold) {
      SDL_SetRenderDrawColor(renderer, 255, 210, 60, 220);
      int prev_y = SPECTRUM_HEIGHT;
      for (int x = 0; x < WINDOW_WIDTH; ++x) {
        float db = peak_db_at(display_peak, x);
        float n = std::clamp((db - MIN_DB) / db_range, 0.0f, 1.0f);
        int y = (int)(SPECTRUM_HEIGHT - n * SPECTRUM_HEIGHT);
        if (x > 0) SDL_RenderDrawLine(renderer, x - 1, prev_y, x, y);
        prev_y = y;
      }
    }

    // Метки dB слева
    if (font) {
      char lbl[16];
      for (int i = 0; i <= DB_STEPS; ++i) {
        float frac = (float)i / DB_STEPS;
        float db = MAX_DB - frac * db_range;
        int y = std::clamp((int)(frac * SPECTRUM_HEIGHT) - 6, 0,
                           SPECTRUM_HEIGHT - 12);
        std::snprintf(lbl, sizeof(lbl), "%.0f", db);
        text_cache.render(lbl, COL_CYAN, 2, y);
      }
    }

    // Маркеры детектированных сигналов
    if (detection_enabled && !signals.empty()) {
      int last_label_x = -1000;
      for (size_t i = 0; i < signals.size(); ++i) {
        const auto &s = signals[i];
        if (s.freq_hz < display_start_freq || s.freq_hz > display_end_freq)
          continue;
        int x = freq_to_px(s.freq_hz);
        float n = std::clamp((s.peak_db - MIN_DB) / db_range, 0.0f, 1.0f);
        int y = (int)(SPECTRUM_HEIGHT - n * SPECTRUM_HEIGHT);
        // Треугольник вершиной вниз над пиком
        SDL_SetRenderDrawColor(renderer, 255, 80, 80, 230);
        for (int dy = 0; dy < 6; ++dy) {
          int dx = 4 - dy * 4 / 6;
          SDL_RenderDrawLine(renderer, x - dx, y - 12 + dy, x + dx,
                             y - 12 + dy);
        }
        // Подпись (избегаем наложения)
        if (font && x - last_label_x >= 80) {
          char t[32];
          format_freq(s.freq_hz, t, sizeof(t));
          int tw, th;
          if (text_cache.measure(t, COL_WHITE, tw, th)) {
            int bx = std::clamp(x - tw / 2, 0, WINDOW_WIDTH - tw);
            int by = std::max(0, y - 14 - th);
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 180);
            SDL_Rect bg{bx - 2, by - 1, tw + 4, th + 2};
            SDL_RenderFillRect(renderer, &bg);
            text_cache.render(t, COL_WHITE, bx, by);
            last_label_x = x;
          }
        }
      }
    }

    // Водопад
    if (waterfall_tex) {
      SDL_Rect r{0, SPECTRUM_HEIGHT, WINDOW_WIDTH, WATERFALL_HEIGHT};
      SDL_RenderCopy(renderer, waterfall_tex, nullptr, &r);
    }

    // Индикатор текущей позиции сканирования
    {
      double cf = current_center_freq.load();
      if (cf >= display_start_freq && cf <= display_end_freq) {
        int cx = freq_to_px(cf);
        int w = std::max(2,
            (int)((double)SAMPLE_RATE / display_total_hz * WINDOW_WIDTH));
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 28);
        SDL_Rect r{cx - w / 2, 0, w, SPECTRUM_HEIGHT + WATERFALL_HEIGHT};
        SDL_RenderFillRect(renderer, &r);
      }
    }

    // PiP
    if (pip_mode.load()) {
      double pc = pip_center_freq.load();
      if (pc >= display_start_freq && pc <= display_end_freq) {
        int x = freq_to_px(pc);
        SDL_SetRenderDrawColor(renderer, 255, 140, 0, 220);
        SDL_RenderDrawLine(renderer, x, 0, x,
                           SPECTRUM_HEIGHT + WATERFALL_HEIGHT);
      }
      if (pip_tex) {
        SDL_Rect r{WINDOW_WIDTH - PIP_WIDTH, 0, PIP_WIDTH, PIP_HEIGHT};
        SDL_RenderCopy(renderer, pip_tex, nullptr, &r);
        SDL_SetRenderDrawColor(renderer, 255, 140, 0, 255);
        SDL_RenderDrawRect(renderer, &r);
        char t[32];
        std::snprintf(t, sizeof(t), "PiP %.3f MHz",
                      pip_center_freq.load() / 1e6);
        text_cache.render(t, COL_ORANGE, WINDOW_WIDTH - PIP_WIDTH + 5,
                          PIP_HEIGHT + 5);
      }
    }

    // Курсор
    SDL_GetMouseState(&mouse_x, &mouse_y);
    if (mouse_x >= 0 && mouse_x < WINDOW_WIDTH && mouse_y >= 0 &&
        mouse_y < SPECTRUM_HEIGHT + WATERFALL_HEIGHT) {
      SDL_SetRenderDrawColor(renderer, 180, 190, 200, 100);
      SDL_RenderDrawLine(renderer, mouse_x, 0, mouse_x,
                         SPECTRUM_HEIGHT + WATERFALL_HEIGHT);
    }

    // Шкала
    SDL_SetRenderDrawColor(renderer, 30, 38, 50, 255);
    SDL_Rect scale_rect{0, WINDOW_HEIGHT - SCALE_HEIGHT, WINDOW_WIDTH,
                        SCALE_HEIGHT};
    SDL_RenderFillRect(renderer, &scale_rect);
    SDL_SetRenderDrawColor(renderer, 60, 70, 85, 255);
    SDL_RenderDrawLine(renderer, 0, WINDOW_HEIGHT - SCALE_HEIGHT, WINDOW_WIDTH,
                       WINDOW_HEIGHT - SCALE_HEIGHT);

    if (font) {
      char t[32];
      for (double mhz = start_r; mhz <= end_r; mhz += grid_step_mhz) {
        int x = freq_to_px(mhz * 1e6);
        if (x < 10 || x > WINDOW_WIDTH - 10) continue;
        SDL_SetRenderDrawColor(renderer, 80, 95, 115, 255);
        SDL_RenderDrawLine(renderer, x, WINDOW_HEIGHT - SCALE_HEIGHT, x,
                           WINDOW_HEIGHT - SCALE_HEIGHT + 5);
        if (grid_step_mhz < 1.0)
          std::snprintf(t, sizeof(t), "%.1f", mhz);
        else
          std::snprintf(t, sizeof(t), "%.0f", mhz);
        int tw, th;
        if (text_cache.measure(t, COL_SCALE_TEXT, tw, th)) {
          int y = WINDOW_HEIGHT - SCALE_HEIGHT + (SCALE_HEIGHT - th) / 2 + 2;
          text_cache.render(t, COL_SCALE_TEXT, x - tw / 2, y);
        }
      }
    }

    if (font) {
      for (const auto &band : bands) {
        double bs = band.start_hz / 1e6;
        double be = band.end_hz / 1e6;
        if (be < start_mhz || bs > end_mhz) continue;
        double is = std::max(bs, start_mhz);
        double ie = std::min(be, end_mhz);
        if (ie <= is) continue;
        double px_w = (ie - is) / display_total_mhz * WINDOW_WIDTH;
        if (px_w < 20.0) continue;
        int x = freq_to_px((is + ie) / 2 * 1e6);
        if (x < 0 || x >= WINDOW_WIDTH) continue;
        int tw, th;
        if (!text_cache.measure(band.name.c_str(), COL_ORANGE, tw, th))
          continue;
        if (x - tw / 2 < 0 || x + tw / 2 > WINDOW_WIDTH) continue;
        int y = WINDOW_HEIGHT - SCALE_HEIGHT - th - 2;
        text_cache.render(band.name.c_str(), COL_ORANGE, x - tw / 2, y);
      }
    }

    // Tooltip: частота + dB (+SNR если есть детекция)
    if (mouse_x >= 0 && mouse_x < WINDOW_WIDTH && mouse_y >= 0 &&
        mouse_y < WINDOW_HEIGHT - SCALE_HEIGHT) {
      double f = display_start_freq +
                 (double)mouse_x / WINDOW_WIDTH * display_total_hz;
      size_t idx = (size_t)std::round((f - SCAN_START_FREQ) / bin_width);
      float db = (idx < total_bins) ? display_scan[idx] : MIN_DB - 100.0f;
      char t[96];
      if (db <= MIN_DB - 99.0f)
        std::snprintf(t, sizeof(t), "%.3f MHz", f / 1e6);
      else if (show_noise_floor || detection_enabled)
        std::snprintf(t, sizeof(t), "%.3f MHz  %.1f dB  SNR %.1f", f / 1e6, db,
                      db - noise_floor_db);
      else
        std::snprintf(t, sizeof(t), "%.3f MHz  %.1f dB", f / 1e6, db);
      int tw, th;
      if (text_cache.measure(t, COL_WHITE, tw, th)) {
        int bx = mouse_x + 10, by = mouse_y + 10;
        if (bx + tw + 6 > WINDOW_WIDTH) bx = mouse_x - 10 - tw - 4;
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 200);
        SDL_Rect bg{bx, by, tw + 4, th + 4};
        SDL_RenderFillRect(renderer, &bg);
        text_cache.render(t, COL_WHITE, bx + 2, by + 2);
      }
    }

    // Info-плашка
    if (font) {
      char info[320];
      std::snprintf(info, sizeof(info),
                    "Disp: %.1f-%.1f MHz  Ctr: %.1f  G: %d  "
                    "NF: %.1f dB  PiP:%s Peak:%s Sm:%s NF:%s Det:%s",
                    start_mhz, end_mhz, current_center_freq.load() / 1e6,
                    sdr_gain.load(), noise_floor_db,
                    pip_mode.load() ? "+" : "-",
                    peak_hold ? "+" : "-",
                    smoothing_enabled ? "+" : "-",
                    show_noise_floor ? "+" : "-",
                    detection_enabled ? "+" : "-");
      int tw = 0, th = 0;
      if (text_cache.measure(info, COL_YELLOW, tw, th)) {
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 160);
        SDL_Rect bg{6, 6, tw + 8, th + 4};
        SDL_RenderFillRect(renderer, &bg);
        text_cache.render(info, COL_YELLOW, 10, 8);
      }
      if (detection_enabled) {
        char c[48];
        std::snprintf(c, sizeof(c), "Signals: %zu", signals.size());
        text_cache.render(c, COL_GREEN, 10, 8 + th + 6);
      }
    }

    SDL_RenderPresent(renderer);
  }

  worker_running = false;
  worker_thread.join();
  if (pip_tex) SDL_DestroyTexture(pip_tex);
  if (waterfall_tex) SDL_DestroyTexture(waterfall_tex);
  text_cache.clear();
  if (font) TTF_CloseFont(font);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(win);
  TTF_Quit();
  SDL_Quit();
  return 0;
}
