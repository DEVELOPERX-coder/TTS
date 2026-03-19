/*
 * encoder.h — Speaker encoder (GRU-based)
 * Voice Cloner TTS System
 */
#ifndef ENCODER_H
#define ENCODER_H

#include "tensor.h"
#include "layers.h"

#define SPEAKER_EMBED_DIM  256
#define SPEAKER_GRU_LAYERS 3
#define SPEAKER_MEL_DIM    80

typedef struct {
    GRUStack   gru;
    LinearLayer projection;  /* hidden_dim → embed_dim */
    int         embed_dim;
} SpeakerEncoder;

/* Create speaker encoder. */
SpeakerEncoder speaker_encoder_create(void);

/* Compute speaker embedding from mel spectrogram.
 * mel: [n_mels, n_frames] → embedding: [embed_dim] (L2-normalized). */
void speaker_encoder_forward(SpeakerEncoder *enc, const Tensor *mel, Tensor *embedding);

/* Compute average embedding from multiple utterances.
 * mels: array of mel spectrograms, n: number of utterances.
 * Returns averaged + L2-normalized embedding [embed_dim]. */
Tensor speaker_encoder_embed_utterances(SpeakerEncoder *enc,
                                        const Tensor *mels, int n);

/* Save/load speaker encoder weights. */
int speaker_encoder_save(const SpeakerEncoder *enc, const char *path);
int speaker_encoder_load(SpeakerEncoder *enc, const char *path);

void speaker_encoder_free(SpeakerEncoder *enc);

#endif /* ENCODER_H */
