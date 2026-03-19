/*
 * vocoder.c — Griffin-Lim vocoder (mel → waveform)
 * Voice Cloner TTS System
 */
#include "vocoder.h"
#include "features.h"
#include "../utils/math_utils.h"
#include "../utils/memory.h"
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════
 * Inverse mel filterbank (pseudo-inverse via transpose approximation)
 * ═══════════════════════════════════════════════════════════════════ */

static void inverse_mel_to_stft(const Tensor *mel_linear, float *stft_mag,
                                int n_bins, int n_frames, int n_mels)
{
    /* Build mel filterbank: [n_mels, n_bins] */
    float *fb = (float *)safe_calloc((size_t)n_mels * (size_t)n_bins, sizeof(float));
    mel_filterbank(fb, SAMPLE_RATE, N_FFT, n_mels, MEL_FMIN, MEL_FMAX);

    /* Transpose and apply: stft_mag[f, t] = sum_m fb[m, f] * mel[m, t] */
    memset(stft_mag, 0, sizeof(float) * (size_t)n_bins * (size_t)n_frames);
    for (int f = 0; f < n_bins; f++) {
        for (int t = 0; t < n_frames; t++) {
            float sum = 0.0f;
            for (int m = 0; m < n_mels; m++) {
                sum += fb[m * n_bins + f] * mel_linear->data[m * n_frames + t];
            }
            stft_mag[f * n_frames + t] = sum > 0.0f ? sum : 0.0f;
        }
    }

    free(fb);
}

/* ═══════════════════════════════════════════════════════════════════
 * Griffin-Lim algorithm
 * ═══════════════════════════════════════════════════════════════════ */

AudioData vocoder_griffin_lim(const Tensor *mel, int sample_rate)
{
    int n_mels   = mel->shape[0];
    int n_frames = mel->shape[1];
    int n_fft    = N_FFT;
    int n_bins   = n_fft / 2 + 1;
    int hop      = HOP_LENGTH;
    int output_len = (n_frames - 1) * hop + n_fft;

    /* Denormalize mel (clone so we don't modify input) */
    Tensor mel_lin = tensor_clone(mel);
    mel_denormalize(&mel_lin);

    /* Invert mel filterbank → approximate STFT magnitude */
    float *target_mag = (float *)safe_calloc((size_t)n_bins * (size_t)n_frames, sizeof(float));
    inverse_mel_to_stft(&mel_lin, target_mag, n_bins, n_frames, n_mels);
    tensor_free(&mel_lin);

    /* Initialize random phase */
    Complexf *stft = (Complexf *)safe_malloc(sizeof(Complexf) * (size_t)n_bins * (size_t)n_frames);
    for (int f = 0; f < n_bins; f++) {
        for (int t = 0; t < n_frames; t++) {
            float phase = uniform(-((float)M_PI), (float)M_PI);
            int idx = f * n_frames + t;
            stft[idx].re = target_mag[idx] * cosf(phase);
            stft[idx].im = target_mag[idx] * sinf(phase);
        }
    }

    /* Allocate workspace */
    float *signal  = (float *)safe_calloc((size_t)output_len, sizeof(float));
    float *window  = (float *)safe_malloc(sizeof(float) * (size_t)n_fft);
    float *win_sum = (float *)safe_calloc((size_t)output_len, sizeof(float));
    hann_window(window, n_fft);

    Complexf *frame_full = (Complexf *)safe_malloc(sizeof(Complexf) * (size_t)n_fft);
    Complexf *frame_bins = (Complexf *)safe_malloc(sizeof(Complexf) * (size_t)n_bins);
    float    *frame_real = (float *)safe_malloc(sizeof(float) * (size_t)n_fft);

    /* Griffin-Lim iterations */
    for (int iter = 0; iter < GRIFFIN_LIM_ITERS; iter++) {
        /* ISTFT: overlap-add */
        memset(signal, 0, sizeof(float) * (size_t)output_len);
        memset(win_sum, 0, sizeof(float) * (size_t)output_len);

        for (int t = 0; t < n_frames; t++) {
            /* Gather frequency bins for this frame */
            for (int f = 0; f < n_bins; f++) {
                frame_bins[f] = stft[f * n_frames + t];
            }

            /* iRFFT */
            irfft(frame_bins, frame_real, n_fft);

            /* Overlap-add */
            int offset = t * hop;
            for (int i = 0; i < n_fft && (offset + i) < output_len; i++) {
                signal[offset + i]  += frame_real[i] * window[i];
                win_sum[offset + i] += window[i] * window[i];
            }
        }

        /* Normalize by window sum */
        for (int i = 0; i < output_len; i++) {
            if (win_sum[i] > 1e-8f) signal[i] /= win_sum[i];
        }

        /* Re-STFT and replace magnitude (keep estimated phase) */
        if (iter < GRIFFIN_LIM_ITERS - 1) {
            float *frame_buf = (float *)safe_calloc((size_t)n_fft, sizeof(float));

            for (int t = 0; t < n_frames; t++) {
                int offset = t * hop;
                memset(frame_buf, 0, sizeof(float) * (size_t)n_fft);
                for (int i = 0; i < n_fft && (offset + i) < output_len; i++) {
                    frame_buf[i] = signal[offset + i] * window[i];
                }

                rfft(frame_buf, frame_bins, n_fft);

                /* Replace magnitude, keep phase */
                for (int f = 0; f < n_bins; f++) {
                    float cur_mag = sqrtf(frame_bins[f].re * frame_bins[f].re +
                                          frame_bins[f].im * frame_bins[f].im);
                    float inv = (cur_mag > 1e-10f)
                                    ? target_mag[f * n_frames + t] / cur_mag
                                    : 0.0f;
                    stft[f * n_frames + t].re = frame_bins[f].re * inv;
                    stft[f * n_frames + t].im = frame_bins[f].im * inv;
                }
            }
            free(frame_buf);
        }
    }

    /* Build output */
    AudioData out;
    out.samples     = signal;
    out.num_samples = output_len;
    out.sample_rate = sample_rate;
    out.channels    = 1;

    /* Cleanup */
    free(target_mag);
    free(stft);
    free(window);
    free(win_sum);
    free(frame_full);
    free(frame_bins);
    free(frame_real);

    return out;
}
