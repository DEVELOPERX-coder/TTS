/*
 * features.c — Audio feature extraction (STFT, Mel spectrogram)
 * Voice Cloner TTS System
 */
#include "features.h"
#include "../utils/math_utils.h"
#include "../utils/memory.h"
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════
 * STFT magnitude
 * ═══════════════════════════════════════════════════════════════════ */

Tensor compute_stft_magnitude(const AudioData *audio)
{
    int n_fft   = N_FFT;
    int hop     = HOP_LENGTH;
    int win_len = WIN_LENGTH;
    int n_bins  = n_fft / 2 + 1;
    int n_frames = (audio->num_samples - win_len) / hop + 1;
    if (n_frames < 1) n_frames = 1;

    /* Build window */
    float *window = (float *)safe_malloc(sizeof(float) * (size_t)win_len);
    hann_window(window, win_len);

    /* Allocate output: [n_bins, n_frames] */
    int shape[2] = { n_bins, n_frames };
    Tensor mag = tensor_zeros(2, shape);

    /* Temp buffers for FFT */
    float    *frame = (float *)safe_calloc((size_t)n_fft, sizeof(float));
    Complexf *spec  = (Complexf *)safe_malloc(sizeof(Complexf) * (size_t)n_bins);

    for (int t = 0; t < n_frames; t++) {
        int offset = t * hop;

        /* Zero-pad and apply window */
        memset(frame, 0, sizeof(float) * (size_t)n_fft);
        for (int i = 0; i < win_len && (offset + i) < audio->num_samples; i++) {
            frame[i] = audio->samples[offset + i] * window[i];
        }

        /* FFT */
        rfft(frame, spec, n_fft);

        /* Magnitude */
        for (int f = 0; f < n_bins; f++) {
            mag.data[f * n_frames + t] = sqrtf(spec[f].re * spec[f].re +
                                                 spec[f].im * spec[f].im);
        }
    }

    free(window);
    free(frame);
    free(spec);

    return mag;
}

/* ═══════════════════════════════════════════════════════════════════
 * Mel filterbank application
 * ═══════════════════════════════════════════════════════════════════ */

Tensor apply_mel_filterbank(const Tensor *stft_mag)
{
    int n_bins   = stft_mag->shape[0];
    int n_frames = stft_mag->shape[1];
    int n_mels   = N_MELS;

    /* Build mel filterbank: [n_mels, n_bins] */
    float *fb = (float *)safe_calloc((size_t)n_mels * (size_t)n_bins, sizeof(float));
    mel_filterbank(fb, SAMPLE_RATE, N_FFT, n_mels, MEL_FMIN, MEL_FMAX);

    /* Output: [n_mels, n_frames] = fb @ stft_mag */
    int out_shape[2] = { n_mels, n_frames };
    Tensor mel = tensor_zeros(2, out_shape);

    for (int m = 0; m < n_mels; m++) {
        for (int t = 0; t < n_frames; t++) {
            float sum = 0.0f;
            for (int f = 0; f < n_bins; f++) {
                sum += fb[m * n_bins + f] * stft_mag->data[f * n_frames + t];
            }
            mel.data[m * n_frames + t] = sum;
        }
    }

    free(fb);
    return mel;
}

/* ═══════════════════════════════════════════════════════════════════
 * Full mel spectrogram pipeline
 * ═══════════════════════════════════════════════════════════════════ */

Tensor compute_mel_spectrogram(const AudioData *audio)
{
    /* 1. STFT magnitude */
    Tensor stft_mag = compute_stft_magnitude(audio);

    /* 2. Apply mel filterbank */
    Tensor mel = apply_mel_filterbank(&stft_mag);
    tensor_free(&stft_mag);

    /* 3. Convert to log scale (dB) */
    for (int i = 0; i < mel.size; i++) {
        mel.data[i] = 20.0f * log10f(mel.data[i] > 1e-10f ? mel.data[i] : 1e-10f);
    }

    /* 4. Normalize */
    mel_normalize(&mel);

    return mel;
}

/* ═══════════════════════════════════════════════════════════════════
 * Normalization
 * ═══════════════════════════════════════════════════════════════════ */

void mel_normalize(Tensor *mel)
{
    /* Clip to [MIN_DB, REF_DB], scale to [0, 1] */
    float range = REF_DB - MIN_DB;
    for (int i = 0; i < mel->size; i++) {
        if (mel->data[i] < MIN_DB) mel->data[i] = MIN_DB;
        if (mel->data[i] > REF_DB)  mel->data[i] = REF_DB;
        mel->data[i] = (mel->data[i] - MIN_DB) / range;
    }
}

void mel_denormalize(Tensor *mel)
{
    float range = REF_DB - MIN_DB;
    for (int i = 0; i < mel->size; i++) {
        mel->data[i] = mel->data[i] * range + MIN_DB;
        /* Convert back from dB */
        mel->data[i] = powf(10.0f, mel->data[i] / 20.0f);
    }
}
