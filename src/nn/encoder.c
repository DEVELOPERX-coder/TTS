/*
 * encoder.c — Speaker encoder (3-layer GRU → L2-normalized embedding)
 * Voice Cloner TTS System
 */
#include "encoder.h"
#include "../utils/math_utils.h"
#include "../utils/memory.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

SpeakerEncoder speaker_encoder_create(void)
{
    SpeakerEncoder enc;
    enc.embed_dim = SPEAKER_EMBED_DIM;
    enc.gru = gru_stack_create(SPEAKER_GRU_LAYERS, SPEAKER_MEL_DIM, SPEAKER_EMBED_DIM);
    enc.projection = linear_create(SPEAKER_EMBED_DIM, SPEAKER_EMBED_DIM);
    return enc;
}

void speaker_encoder_forward(SpeakerEncoder *enc, const Tensor *mel, Tensor *embedding)
{
    int n_mels   = mel->shape[0];
    int n_frames = mel->shape[1];

    /* Transpose mel: [n_mels, n_frames] → [n_frames, n_mels] for GRU input */
    int input_shape[2] = { n_frames, n_mels };
    Tensor input = tensor_zeros(2, input_shape);
    for (int t = 0; t < n_frames; t++) {
        for (int m = 0; m < n_mels; m++) {
            input.data[t * n_mels + m] = mel->data[m * n_frames + t];
        }
    }

    /* Run GRU stack */
    int out_shape[2] = { n_frames, SPEAKER_EMBED_DIM };
    Tensor gru_out = tensor_zeros(2, out_shape);
    int h_shape[1] = { SPEAKER_EMBED_DIM };
    Tensor final_h = tensor_zeros(1, h_shape);

    gru_stack_forward(&enc->gru, &input, &gru_out, &final_h);

    /* Project final hidden state */
    int e_shape[1] = { SPEAKER_EMBED_DIM };
    Tensor proj = tensor_zeros(1, e_shape);
    linear_forward(&enc->projection, &final_h, &proj);

    /* ReLU activation */
    for (int i = 0; i < SPEAKER_EMBED_DIM; i++) {
        proj.data[i] = relu_f(proj.data[i]);
    }

    /* L2 normalize */
    float norm = tensor_l2_norm(&proj);
    if (norm > 1e-10f) {
        for (int i = 0; i < SPEAKER_EMBED_DIM; i++) {
            embedding->data[i] = proj.data[i] / norm;
        }
    } else {
        memcpy(embedding->data, proj.data, sizeof(float) * SPEAKER_EMBED_DIM);
    }

    tensor_free(&input);
    tensor_free(&gru_out);
    tensor_free(&final_h);
    tensor_free(&proj);
}

Tensor speaker_encoder_embed_utterances(SpeakerEncoder *enc,
                                        const Tensor *mels, int n)
{
    int e_shape[1] = { SPEAKER_EMBED_DIM };
    Tensor avg = tensor_zeros(1, e_shape);
    Tensor emb = tensor_zeros(1, e_shape);

    for (int i = 0; i < n; i++) {
        speaker_encoder_forward(enc, &mels[i], &emb);
        for (int j = 0; j < SPEAKER_EMBED_DIM; j++) {
            avg.data[j] += emb.data[j];
        }
    }

    /* Average */
    if (n > 0) {
        float inv_n = 1.0f / (float)n;
        for (int j = 0; j < SPEAKER_EMBED_DIM; j++) {
            avg.data[j] *= inv_n;
        }
    }

    /* L2 normalize the average */
    float norm = tensor_l2_norm(&avg);
    if (norm > 1e-10f) {
        for (int j = 0; j < SPEAKER_EMBED_DIM; j++) {
            avg.data[j] /= norm;
        }
    }

    tensor_free(&emb);
    return avg;
}

/* ═══════════════════════════════════════════════════════════════════
 * Save/Load
 * ═══════════════════════════════════════════════════════════════════ */

static int save_tensor(FILE *f, const Tensor *t)
{
    fwrite(&t->ndim, sizeof(int), 1, f);
    fwrite(t->shape, sizeof(int), (size_t)t->ndim, f);
    fwrite(&t->size, sizeof(int), 1, f);
    fwrite(t->data, sizeof(float), (size_t)t->size, f);
    return 0;
}

static int load_tensor(FILE *f, Tensor *t)
{
    int ndim;
    if (fread(&ndim, sizeof(int), 1, f) != 1) return -1;
    int shape[TENSOR_MAX_DIMS];
    if (fread(shape, sizeof(int), (size_t)ndim, f) != (size_t)ndim) return -1;

    int size;
    if (fread(&size, sizeof(int), 1, f) != 1) return -1;

    /* Recreate tensor if needed */
    if (t->data) tensor_free(t);
    *t = tensor_create(ndim, shape);
    if (fread(t->data, sizeof(float), (size_t)size, f) != (size_t)size) return -1;
    return 0;
}

static void save_linear(FILE *f, const LinearLayer *l)
{
    save_tensor(f, &l->W);
    save_tensor(f, &l->b);
}

static void load_linear(FILE *f, LinearLayer *l)
{
    load_tensor(f, &l->W);
    load_tensor(f, &l->b);
}

static void save_gru_cell(FILE *f, const GRUCell *g)
{
    save_linear(f, &g->W_z); save_linear(f, &g->U_z);
    save_linear(f, &g->W_r); save_linear(f, &g->U_r);
    save_linear(f, &g->W_h); save_linear(f, &g->U_h);
}

static void load_gru_cell(FILE *f, GRUCell *g)
{
    load_linear(f, &g->W_z); load_linear(f, &g->U_z);
    load_linear(f, &g->W_r); load_linear(f, &g->U_r);
    load_linear(f, &g->W_h); load_linear(f, &g->U_h);
}

int speaker_encoder_save(const SpeakerEncoder *enc, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Magic + version */
    int magic = 0x53504B45; /* "SPKE" */
    int version = 1;
    fwrite(&magic, sizeof(int), 1, f);
    fwrite(&version, sizeof(int), 1, f);

    /* GRU layers */
    for (int i = 0; i < enc->gru.n_layers; i++) {
        save_gru_cell(f, &enc->gru.cells[i]);
    }
    save_linear(f, &enc->projection);

    fclose(f);
    printf("[INFO] Speaker encoder saved to %s\n", path);
    return 0;
}

int speaker_encoder_load(SpeakerEncoder *enc, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    int magic, version;
    if (fread(&magic, sizeof(int), 1, f) != 1 || magic != 0x53504B45) {
        fclose(f);
        return -1;
    }
    fread(&version, sizeof(int), 1, f);

    for (int i = 0; i < enc->gru.n_layers; i++) {
        load_gru_cell(f, &enc->gru.cells[i]);
    }
    load_linear(f, &enc->projection);

    fclose(f);
    printf("[INFO] Speaker encoder loaded from %s\n", path);
    return 0;
}

void speaker_encoder_free(SpeakerEncoder *enc)
{
    gru_stack_free(&enc->gru);
    linear_free(&enc->projection);
}
