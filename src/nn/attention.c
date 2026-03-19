/*
 * attention.c — Location-sensitive attention (Chorowski et al.)
 * Voice Cloner TTS System
 */
#include "attention.h"
#include "../utils/math_utils.h"
#include "../utils/memory.h"
#include <math.h>
#include <string.h>

LocationSensitiveAttention attention_create(int enc_dim, int dec_dim, int attn_dim)
{
    LocationSensitiveAttention a;
    a.attn_dim = attn_dim;
    a.enc_dim  = enc_dim;
    a.dec_dim  = dec_dim;

    a.query_layer    = linear_create(dec_dim, attn_dim);
    a.memory_layer   = linear_create(enc_dim, attn_dim);
    a.v_layer        = linear_create(attn_dim, 1);
    a.location_conv  = conv1d_create(1, ATTN_FILTERS, ATTN_KERNEL);
    a.location_layer = linear_create(ATTN_FILTERS, attn_dim);

    return a;
}

void attention_forward(LocationSensitiveAttention *attn,
                       const Tensor *query,
                       const Tensor *memory,
                       const Tensor *cum_attn,
                       Tensor *attn_out,
                       Tensor *context)
{
    int enc_len  = memory->shape[0];
    int enc_dim  = attn->enc_dim;
    int attn_dim = attn->attn_dim;

    /* 1. Process query: [dec_dim] → [attn_dim] */
    int q_shape[1] = { attn_dim };
    Tensor q_proj = tensor_zeros(1, q_shape);
    linear_forward(&attn->query_layer, query, &q_proj);

    /* 2. Process memory: [enc_len, enc_dim] → [enc_len, attn_dim] */
    int m_shape[2] = { enc_len, attn_dim };
    Tensor m_proj = tensor_zeros(2, m_shape);
    linear_forward(&attn->memory_layer, memory, &m_proj);

    /* 3. Process location: cum_attn [enc_len] → [1, enc_len] → conv → [filters, enc_len] → [enc_len, filters] → linear → [enc_len, attn_dim] */
    int loc_in_shape[2] = { 1, enc_len };
    Tensor loc_in = tensor_zeros(2, loc_in_shape);
    memcpy(loc_in.data, cum_attn->data, sizeof(float) * (size_t)enc_len);

    int loc_conv_shape[2] = { ATTN_FILTERS, enc_len };
    Tensor loc_conv_out = tensor_zeros(2, loc_conv_shape);
    conv1d_forward(&attn->location_conv, &loc_in, &loc_conv_out);

    /* Transpose to [enc_len, ATTN_FILTERS] for linear layer */
    int loc_t_shape[2] = { enc_len, ATTN_FILTERS };
    Tensor loc_t = tensor_zeros(2, loc_t_shape);
    for (int t = 0; t < enc_len; t++) {
        for (int f = 0; f < ATTN_FILTERS; f++) {
            loc_t.data[t * ATTN_FILTERS + f] = loc_conv_out.data[f * enc_len + t];
        }
    }

    int loc_proj_shape[2] = { enc_len, attn_dim };
    Tensor loc_proj = tensor_zeros(2, loc_proj_shape);
    linear_forward(&attn->location_layer, &loc_t, &loc_proj);

    /* 4. Energy: e_i = v^T · tanh(query_proj + memory_proj[i] + location_proj[i]) */
    float *energies = (float *)safe_malloc(sizeof(float) * (size_t)enc_len);
    for (int i = 0; i < enc_len; i++) {
        /* Compute tanh sum for this timestep */
        int one_shape[2] = { 1, attn_dim };
        Tensor tanh_in = tensor_zeros(2, one_shape);
        for (int d = 0; d < attn_dim; d++) {
            tanh_in.data[d] = tanh_f(q_proj.data[d] +
                                      m_proj.data[i * attn_dim + d] +
                                      loc_proj.data[i * attn_dim + d]);
        }

        int e_shape[2] = { 1, 1 };
        Tensor e = tensor_zeros(2, e_shape);
        linear_forward(&attn->v_layer, &tanh_in, &e);
        energies[i] = e.data[0];
        tensor_free(&tanh_in);
        tensor_free(&e);
    }

    /* 5. Softmax to get attention weights */
    float max_e = energies[0];
    for (int i = 1; i < enc_len; i++)
        if (energies[i] > max_e) max_e = energies[i];

    float sum_exp = 0.0f;
    for (int i = 0; i < enc_len; i++) {
        attn_out->data[i] = expf(energies[i] - max_e);
        sum_exp += attn_out->data[i];
    }
    for (int i = 0; i < enc_len; i++) {
        attn_out->data[i] /= (sum_exp + 1e-10f);
    }

    /* 6. Context vector: weighted sum of memory */
    memset(context->data, 0, sizeof(float) * (size_t)enc_dim);
    for (int i = 0; i < enc_len; i++) {
        float w = attn_out->data[i];
        for (int d = 0; d < enc_dim; d++) {
            context->data[d] += w * memory->data[i * enc_dim + d];
        }
    }

    /* Cleanup */
    free(energies);
    tensor_free(&q_proj);
    tensor_free(&m_proj);
    tensor_free(&loc_in);
    tensor_free(&loc_conv_out);
    tensor_free(&loc_t);
    tensor_free(&loc_proj);
}

void attention_free(LocationSensitiveAttention *attn)
{
    linear_free(&attn->query_layer);
    linear_free(&attn->memory_layer);
    linear_free(&attn->v_layer);
    conv1d_free(&attn->location_conv);
    linear_free(&attn->location_layer);
}
