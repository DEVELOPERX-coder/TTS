/*
 * layers.h — Neural network layers
 * Voice Cloner TTS System
 */
#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"

/* ── Linear layer ───────────────────────────────────────────────── */
typedef struct {
    Tensor W;       /* [out_dim, in_dim] */
    Tensor b;       /* [out_dim] */
    Tensor dW;      /* gradients */
    Tensor db;
    int    in_dim;
    int    out_dim;
} LinearLayer;

LinearLayer linear_create(int in_dim, int out_dim);
void linear_forward(LinearLayer *l, const Tensor *input, Tensor *output);
void linear_free(LinearLayer *l);

/* ── GRU cell ───────────────────────────────────────────────────── */
typedef struct {
    LinearLayer W_z, W_r, W_h;   /* input → gates */
    LinearLayer U_z, U_r, U_h;   /* hidden → gates */
    int input_dim;
    int hidden_dim;
} GRUCell;

GRUCell gru_create(int input_dim, int hidden_dim);
void gru_forward(GRUCell *g, const Tensor *x, const Tensor *h_prev, Tensor *h_next);
void gru_free(GRUCell *g);

/* ── Multi-layer GRU ────────────────────────────────────────────── */
typedef struct {
    GRUCell *cells;
    int      n_layers;
    int      input_dim;
    int      hidden_dim;
} GRUStack;

GRUStack gru_stack_create(int n_layers, int input_dim, int hidden_dim);
/* Process full sequence. input: [seq_len, input_dim].
 * output: [seq_len, hidden_dim]. final_h: [hidden_dim]. */
void gru_stack_forward(GRUStack *g, const Tensor *input, Tensor *output, Tensor *final_h);
void gru_stack_free(GRUStack *g);

/* ── Conv1D layer ───────────────────────────────────────────────── */
typedef struct {
    Tensor W;        /* [out_ch, in_ch, kernel] */
    Tensor b;        /* [out_ch] */
    Tensor dW, db;
    int    in_ch;
    int    out_ch;
    int    kernel;
    int    padding;  /* same-padding size */
} Conv1DLayer;

Conv1DLayer conv1d_create(int in_ch, int out_ch, int kernel);
/* input: [in_ch, seq_len], output: [out_ch, seq_len] */
void conv1d_forward(Conv1DLayer *l, const Tensor *input, Tensor *output);
void conv1d_free(Conv1DLayer *l);

/* ── Embedding layer ────────────────────────────────────────────── */
typedef struct {
    Tensor W;       /* [vocab_size, embed_dim] */
    Tensor dW;
    int    vocab_size;
    int    embed_dim;
} EmbeddingLayer;

EmbeddingLayer embedding_create(int vocab_size, int embed_dim);
/* input: [seq_len] int tokens, output: [seq_len, embed_dim] */
void embedding_forward(EmbeddingLayer *e, const int *tokens, int n, Tensor *output);
void embedding_free(EmbeddingLayer *e);

/* ── LayerNorm ──────────────────────────────────────────────────── */
typedef struct {
    Tensor gamma;    /* [dim] */
    Tensor beta;     /* [dim] */
    int    dim;
    float  eps;
} LayerNormLayer;

LayerNormLayer layer_norm_create(int dim);
void layer_norm_forward(LayerNormLayer *ln, const Tensor *input, Tensor *output);
void layer_norm_free(LayerNormLayer *ln);

/* ── Prenet (Tacotron decoder prenet) ───────────────────────────── */
typedef struct {
    LinearLayer fc1;
    LinearLayer fc2;
    int input_dim;
    int hidden_dim;
    float dropout;
} PrenetLayer;

PrenetLayer prenet_create(int input_dim, int hidden_dim, float dropout);
void prenet_forward(PrenetLayer *p, const Tensor *input, Tensor *output, int training);
void prenet_free(PrenetLayer *p);

/* ── PostNet (5x Conv1D bank) ───────────────────────────────────── */
#define POSTNET_CONVS 5
#define POSTNET_CH    512
#define POSTNET_KERNEL 5

typedef struct {
    Conv1DLayer convs[POSTNET_CONVS];
    int n_mels;
} PostnetLayer;

PostnetLayer postnet_create(int n_mels);
/* input/output: [n_mels, n_frames] — residual connection */
void postnet_forward(PostnetLayer *p, const Tensor *input, Tensor *output);
void postnet_free(PostnetLayer *p);

/* ── He initialization ──────────────────────────────────────────── */
void he_init(Tensor *W, int fan_in);

#endif /* LAYERS_H */
