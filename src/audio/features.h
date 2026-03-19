/*
 * features.h — Audio feature extraction (STFT, Mel spectrogram)
 * Voice Cloner TTS System
 */
#ifndef FEATURES_H
#define FEATURES_H

#include "wav.h"
#include "../nn/tensor.h"

/* Default audio parameters */
#define SAMPLE_RATE     16000
#define N_FFT           1024
#define HOP_LENGTH      256
#define WIN_LENGTH      1024
#define N_MELS          80
#define MEL_FMIN        0.0f
#define MEL_FMAX        8000.0f
#define REF_DB          20.0f
#define MIN_DB          (-100.0f)

/* Compute log-mel spectrogram from audio.
 * Returns Tensor of shape [n_mels, n_frames]. */
Tensor compute_mel_spectrogram(const AudioData *audio);

/* Compute STFT magnitude from audio.
 * Returns Tensor of shape [n_fft/2+1, n_frames]. */
Tensor compute_stft_magnitude(const AudioData *audio);

/* Apply mel filterbank to STFT magnitude.
 * stft_mag: [n_fft/2+1, n_frames]
 * Returns: [n_mels, n_frames] */
Tensor apply_mel_filterbank(const Tensor *stft_mag);

/* Normalize mel spectrogram to [0, 1] range. */
void mel_normalize(Tensor *mel);

/* Denormalize mel spectrogram from [0, 1] range. */
void mel_denormalize(Tensor *mel);

#endif /* FEATURES_H */
