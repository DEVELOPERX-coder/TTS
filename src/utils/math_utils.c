/*
 * math_utils.c — Mathematical utilities (FFT, windowing, activations)
 * Voice Cloner TTS System
 */
#include "math_utils.h"
#include "memory.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ═══════════════════════════════════════════════════════════════════
 * FFT — Cooley-Tukey radix-2, in-place
 * ═══════════════════════════════════════════════════════════════════ */

static void bit_reverse(Complexf *x, int n)
{
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            Complexf tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
    }
}

void fft(Complexf *x, int n, int inverse)
{
    bit_reverse(x, n);

    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2.0f * (float)M_PI / (float)len * (inverse ? -1.0f : 1.0f);
        Complexf wlen = { cosf(ang), sinf(ang) };

        for (int i = 0; i < n; i += len) {
            Complexf w = { 1.0f, 0.0f };
            for (int j = 0; j < len / 2; j++) {
                Complexf u = x[i + j];
                Complexf v = {
                    w.re * x[i + j + len/2].re - w.im * x[i + j + len/2].im,
                    w.re * x[i + j + len/2].im + w.im * x[i + j + len/2].re
                };
                x[i + j].re = u.re + v.re;
                x[i + j].im = u.im + v.im;
                x[i + j + len/2].re = u.re - v.re;
                x[i + j + len/2].im = u.im - v.im;

                float wr = w.re * wlen.re - w.im * wlen.im;
                float wi = w.re * wlen.im + w.im * wlen.re;
                w.re = wr;
                w.im = wi;
            }
        }
    }

    if (inverse) {
        for (int i = 0; i < n; i++) {
            x[i].re /= (float)n;
            x[i].im /= (float)n;
        }
    }
}

void rfft(const float *input, Complexf *output, int n)
{
    Complexf *buf = (Complexf *)safe_malloc(sizeof(Complexf) * (size_t)n);
    for (int i = 0; i < n; i++) {
        buf[i].re = input[i];
        buf[i].im = 0.0f;
    }
    fft(buf, n, 0);
    /* Copy first n/2+1 bins */
    memcpy(output, buf, sizeof(Complexf) * (size_t)(n / 2 + 1));
    free(buf);
}

void irfft(const Complexf *input, float *output, int n)
{
    Complexf *buf = (Complexf *)safe_malloc(sizeof(Complexf) * (size_t)n);
    /* Copy first n/2+1 bins */
    for (int i = 0; i <= n / 2; i++) {
        buf[i] = input[i];
    }
    /* Mirror conjugate */
    for (int i = n / 2 + 1; i < n; i++) {
        buf[i].re =  buf[n - i].re;
        buf[i].im = -buf[n - i].im;
    }
    fft(buf, n, 1);
    for (int i = 0; i < n; i++) {
        output[i] = buf[i].re;
    }
    free(buf);
}

/* ═══════════════════════════════════════════════════════════════════
 * Windowing
 * ═══════════════════════════════════════════════════════════════════ */

void hann_window(float *w, int n)
{
    for (int i = 0; i < n; i++) {
        w[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)(n - 1)));
    }
}

void hamming_window(float *w, int n)
{
    for (int i = 0; i < n; i++) {
        w[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * (float)i / (float)(n - 1));
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Mel filterbank
 * ═══════════════════════════════════════════════════════════════════ */

float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void mel_filterbank(float *out, int sr, int n_fft, int n_mels,
                    float fmin, float fmax)
{
    int n_bins = n_fft / 2 + 1;
    if (fmax <= 0.0f) fmax = (float)sr / 2.0f;

    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);

    /* n_mels + 2 center frequencies in mel scale */
    float *mel_pts = (float *)safe_malloc(sizeof(float) * (size_t)(n_mels + 2));
    for (int i = 0; i < n_mels + 2; i++) {
        float mel = mel_min + (mel_max - mel_min) * (float)i / (float)(n_mels + 1);
        mel_pts[i] = mel_to_hz(mel);
    }

    /* Convert Hz to FFT bin */
    float *bin_pts = (float *)safe_malloc(sizeof(float) * (size_t)(n_mels + 2));
    for (int i = 0; i < n_mels + 2; i++) {
        bin_pts[i] = mel_pts[i] * (float)n_fft / (float)sr;
    }

    /* Zero the output */
    memset(out, 0, sizeof(float) * (size_t)n_mels * (size_t)n_bins);

    /* Build triangular filters */
    for (int m = 0; m < n_mels; m++) {
        float f_left   = bin_pts[m];
        float f_center = bin_pts[m + 1];
        float f_right  = bin_pts[m + 2];

        for (int k = 0; k < n_bins; k++) {
            float fk = (float)k;
            if (fk >= f_left && fk <= f_center) {
                out[m * n_bins + k] = (fk - f_left) / (f_center - f_left + 1e-10f);
            } else if (fk > f_center && fk <= f_right) {
                out[m * n_bins + k] = (f_right - fk) / (f_right - f_center + 1e-10f);
            }
        }
    }

    free(mel_pts);
    free(bin_pts);
}

/* ═══════════════════════════════════════════════════════════════════
 * Random number generation
 * ═══════════════════════════════════════════════════════════════════ */

static int g_rng_seeded = 0;

static void ensure_seeded(void)
{
    if (!g_rng_seeded) {
        srand((unsigned)time(NULL));
        g_rng_seeded = 1;
    }
}

float randn(void)
{
    ensure_seeded();
    /* Box-Muller transform */
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

float uniform(float lo, float hi)
{
    ensure_seeded();
    float r = (float)rand() / (float)RAND_MAX;
    return lo + r * (hi - lo);
}
