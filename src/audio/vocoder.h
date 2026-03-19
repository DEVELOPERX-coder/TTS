/*
 * vocoder.h — Griffin-Lim vocoder (mel → waveform)
 * Voice Cloner TTS System
 */
#ifndef VOCODER_H
#define VOCODER_H

#include "../nn/tensor.h"
#include "wav.h"

#define GRIFFIN_LIM_ITERS 60

/* Convert mel spectrogram back to audio waveform using Griffin-Lim.
 * mel: [n_mels, n_frames], should be in normalized [0,1] range.
 * Returns AudioData at SAMPLE_RATE. */
AudioData vocoder_griffin_lim(const Tensor *mel, int sample_rate);

#endif /* VOCODER_H */
