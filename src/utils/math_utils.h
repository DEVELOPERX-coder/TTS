/*
 * math_utils.h — Mathematical utilities (FFT, windowing, activations)
 * Voice Cloner TTS System
 */
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Complex number ─────────────────────────────────────────────── */
typedef struct { float re, im; } Complexf;

/* ── FFT ────────────────────────────────────────────────────────── */

/* In-place radix-2 Cooley-Tukey FFT.  n must be power of 2. */
void fft(Complexf *x, int n, int inverse);

/* Real-valued FFT.  Input: n real samples.  Output: n/2+1 complex bins. */
void rfft(const float *input, Complexf *output, int n);

/* Inverse real FFT.  Input: n/2+1 complex bins.  Output: n real samples. */
void irfft(const Complexf *input, float *output, int n);

/* ── Windowing ──────────────────────────────────────────────────── */

void hann_window(float *w, int n);
void hamming_window(float *w, int n);

/* ── Mel filterbank ─────────────────────────────────────────────── */

/* Build triangular mel filterbank matrix.
 * out: [n_mels x (n_fft/2+1)] stored row-major.
 * sr: sample rate, n_fft: FFT size, n_mels: number of mel bands.
 */
void mel_filterbank(float *out, int sr, int n_fft, int n_mels,
                    float fmin, float fmax);

/* Hz ↔ Mel conversions */
float hz_to_mel(float hz);
float mel_to_hz(float mel);

/* ── Activation functions ───────────────────────────────────────── */

static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}
static inline float tanh_f(float x) {
    return tanhf(x);
}
static inline float relu_f(float x) {
    return x > 0.0f ? x : 0.0f;
}
static inline float log_f(float x) {
    return logf(x > 1e-10f ? x : 1e-10f);
}

/* ── Random ─────────────────────────────────────────────────────── */
float randn(void);  /* standard normal N(0,1) using Box-Muller */
float uniform(float lo, float hi);

#endif /* MATH_UTILS_H */
