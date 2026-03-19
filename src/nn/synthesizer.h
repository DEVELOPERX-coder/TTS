/*
 * synthesizer.h — Tacotron-lite TTS Synthesizer
 * Voice Cloner TTS System
 */
#ifndef SYNTHESIZER_H
#define SYNTHESIZER_H

#include "tensor.h"
#include "layers.h"
#include "attention.h"

/* Synthesizer hyperparameters */
#define SYN_EMBED_DIM   256
#define SYN_ENC_DIM     256
#define SYN_DEC_DIM     512
#define SYN_ATTN_DIM    128
#define SYN_PRENET_DIM  128
#define SYN_N_MELS      80
#define SYN_MAX_FRAMES  1000

typedef struct {
    /* Text encoder */
    EmbeddingLayer  text_embed;
    Conv1DLayer     enc_conv[3];
    GRUStack        enc_gru;

    /* Decoder */
    PrenetLayer     prenet;
    GRUCell         attn_gru;
    GRUCell         dec_gru;
    LocationSensitiveAttention attention;

    /* Output projections */
    LinearLayer     mel_proj;       /* → n_mels */
    LinearLayer     stop_proj;      /* → 1 (stop token) */
    PostnetLayer    postnet;

    /* Speaker conditioning */
    LinearLayer     speaker_proj;   /* speaker_embed → enc_dim */

    int vocab_size;
} TTSSynthesizer;

/* Create synthesizer. */
TTSSynthesizer synthesizer_create(int vocab_size);

/* Synthesize mel spectrogram from text tokens and speaker embedding.
 * tokens:    int array of length n_tokens
 * speaker:   [SPEAKER_EMBED_DIM]
 * mel_out:   output Tensor [n_mels, n_frames] (allocated by function)
 * Returns number of frames generated. */
int synthesizer_inference(TTSSynthesizer *syn,
                          const int *tokens, int n_tokens,
                          const Tensor *speaker_embed,
                          Tensor *mel_out);

/* Run one training step with teacher forcing.
 * tokens:     input text tokens
 * n_tokens:   number of tokens
 * speaker:    [SPEAKER_EMBED_DIM]
 * target_mel: [n_mels, n_frames] target mel spectrogram
 * Returns loss value. */
float synthesizer_train_step(TTSSynthesizer *syn,
                             const int *tokens, int n_tokens,
                             const Tensor *speaker_embed,
                             const Tensor *target_mel);

/* Save/load synthesizer weights. */
int synthesizer_save(const TTSSynthesizer *syn, const char *path);
int synthesizer_load(TTSSynthesizer *syn, const char *path);

void synthesizer_free(TTSSynthesizer *syn);

#endif /* SYNTHESIZER_H */
