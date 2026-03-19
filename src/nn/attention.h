/*
 * attention.h — Location-sensitive attention
 * Voice Cloner TTS System
 */
#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"
#include "layers.h"

#define ATTN_FILTERS    32
#define ATTN_KERNEL     31

typedef struct {
    LinearLayer query_layer;    /* decoder state → attention dim */
    LinearLayer memory_layer;   /* encoder output → attention dim */
    LinearLayer v_layer;        /* energy → scalar */
    Conv1DLayer location_conv;  /* cumulative attention → features */
    LinearLayer location_layer; /* location features → attention dim */
    int         attn_dim;
    int         enc_dim;
    int         dec_dim;
} LocationSensitiveAttention;

LocationSensitiveAttention attention_create(int enc_dim, int dec_dim, int attn_dim);

/* Compute attention weights and context vector.
 * query:     [dec_dim]         — decoder hidden state
 * memory:    [enc_len, enc_dim] — encoder outputs
 * cum_attn:  [enc_len]         — cumulative attention weights
 * attn_out:  [enc_len]         — output attention weights
 * context:   [enc_dim]         — output context vector */
void attention_forward(LocationSensitiveAttention *attn,
                       const Tensor *query,
                       const Tensor *memory,
                       const Tensor *cum_attn,
                       Tensor *attn_out,
                       Tensor *context);

void attention_free(LocationSensitiveAttention *attn);

#endif /* ATTENTION_H */
