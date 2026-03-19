/*
 * tensor.c — N-dimensional tensor operations
 * Voice Cloner TTS System
 */
#include "tensor.h"
#include "../utils/math_utils.h"
#include "../utils/memory.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════
 * Lifecycle
 * ═══════════════════════════════════════════════════════════════════ */

static int compute_size(int ndim, const int *shape)
{
    int s = 1;
    for (int i = 0; i < ndim; i++) s *= shape[i];
    return s;
}

Tensor tensor_create(int ndim, const int *shape)
{
    Tensor t;
    t.ndim = ndim;
    t.size = compute_size(ndim, shape);
    memcpy(t.shape, shape, sizeof(int) * (size_t)ndim);
    for (int i = ndim; i < TENSOR_MAX_DIMS; i++) t.shape[i] = 0;
    t.data = (float *)safe_malloc(sizeof(float) * (size_t)t.size);
    t.owns_data = 1;
    return t;
}

Tensor tensor_zeros(int ndim, const int *shape)
{
    Tensor t = tensor_create(ndim, shape);
    memset(t.data, 0, sizeof(float) * (size_t)t.size);
    return t;
}

Tensor tensor_ones(int ndim, const int *shape)
{
    Tensor t = tensor_create(ndim, shape);
    for (int i = 0; i < t.size; i++) t.data[i] = 1.0f;
    return t;
}

Tensor tensor_randn(int ndim, const int *shape)
{
    Tensor t = tensor_create(ndim, shape);
    for (int i = 0; i < t.size; i++) t.data[i] = randn();
    return t;
}

Tensor tensor_clone(const Tensor *t)
{
    Tensor c = tensor_create(t->ndim, t->shape);
    memcpy(c.data, t->data, sizeof(float) * (size_t)t->size);
    return c;
}

void tensor_free(Tensor *t)
{
    if (t->owns_data && t->data) {
        free(t->data);
    }
    t->data = NULL;
    t->size = 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Views
 * ═══════════════════════════════════════════════════════════════════ */

Tensor tensor_view(Tensor *t, int ndim, const int *shape)
{
    Tensor v;
    v.ndim = ndim;
    v.size = compute_size(ndim, shape);
    memcpy(v.shape, shape, sizeof(int) * (size_t)ndim);
    for (int i = ndim; i < TENSOR_MAX_DIMS; i++) v.shape[i] = 0;
    v.data = t->data;
    v.owns_data = 0;
    return v;
}

/* ═══════════════════════════════════════════════════════════════════
 * Element-wise ops
 * ═══════════════════════════════════════════════════════════════════ */

void tensor_add(Tensor *dst, const Tensor *a, const Tensor *b)
{
    for (int i = 0; i < dst->size; i++)
        dst->data[i] = a->data[i] + b->data[i];
}

void tensor_add_inplace(Tensor *dst, const Tensor *src)
{
    for (int i = 0; i < dst->size; i++)
        dst->data[i] += src->data[i];
}

void tensor_sub(Tensor *dst, const Tensor *a, const Tensor *b)
{
    for (int i = 0; i < dst->size; i++)
        dst->data[i] = a->data[i] - b->data[i];
}

void tensor_mul_elem(Tensor *dst, const Tensor *a, const Tensor *b)
{
    for (int i = 0; i < dst->size; i++)
        dst->data[i] = a->data[i] * b->data[i];
}

void tensor_scale(Tensor *t, float s)
{
    for (int i = 0; i < t->size; i++)
        t->data[i] *= s;
}

void tensor_fill(Tensor *t, float val)
{
    for (int i = 0; i < t->size; i++)
        t->data[i] = val;
}

/* ═══════════════════════════════════════════════════════════════════
 * Matrix ops
 * ═══════════════════════════════════════════════════════════════════ */

void tensor_matmul(Tensor *C, const Tensor *A, const Tensor *B)
{
    int M = A->shape[0], K = A->shape[1], N = B->shape[1];
    memset(C->data, 0, sizeof(float) * (size_t)(M * N));

    /* Loop-tiled for cache efficiency */
    #define TILE 32
    for (int i0 = 0; i0 < M; i0 += TILE)
    for (int j0 = 0; j0 < N; j0 += TILE)
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int imax = i0 + TILE < M ? i0 + TILE : M;
        int jmax = j0 + TILE < N ? j0 + TILE : N;
        int kmax = k0 + TILE < K ? k0 + TILE : K;
        for (int i = i0; i < imax; i++)
        for (int k = k0; k < kmax; k++) {
            float a_ik = A->data[i * K + k];
            for (int j = j0; j < jmax; j++)
                C->data[i * N + j] += a_ik * B->data[k * N + j];
        }
    }
    #undef TILE
}

void tensor_matmul_transA(Tensor *C, const Tensor *A, const Tensor *B)
{
    /* A^T @ B :  A:[K,M] → A^T:[M,K],  B:[K,N],  C:[M,N] */
    int K = A->shape[0], M = A->shape[1], N = B->shape[1];
    memset(C->data, 0, sizeof(float) * (size_t)(M * N));
    for (int k = 0; k < K; k++)
    for (int i = 0; i < M; i++) {
        float a_ki = A->data[k * M + i];
        for (int j = 0; j < N; j++)
            C->data[i * N + j] += a_ki * B->data[k * N + j];
    }
}

void tensor_matmul_transB(Tensor *C, const Tensor *A, const Tensor *B)
{
    /* A @ B^T :  A:[M,K],  B:[N,K] → B^T:[K,N],  C:[M,N] */
    int M = A->shape[0], K = A->shape[1], N = B->shape[0];
    memset(C->data, 0, sizeof(float) * (size_t)(M * N));
    for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A->data[i * K + k] * B->data[j * K + k];
        C->data[i * N + j] = sum;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Reductions
 * ═══════════════════════════════════════════════════════════════════ */

void tensor_softmax(Tensor *t, int axis)
{
    (void)axis; /* simplified: last axis only, 2D [rows, cols] */
    int rows = t->shape[0];
    int cols = t->size / rows;
    for (int r = 0; r < rows; r++) {
        float *row = t->data + r * cols;
        float mx = row[0];
        for (int c = 1; c < cols; c++)
            if (row[c] > mx) mx = row[c];
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - mx);
            sum += row[c];
        }
        float inv = 1.0f / (sum + 1e-10f);
        for (int c = 0; c < cols; c++)
            row[c] *= inv;
    }
}

void tensor_layer_norm(Tensor *out, const Tensor *in, const Tensor *gamma,
                       const Tensor *beta, float eps)
{
    int last = in->shape[in->ndim - 1];
    int outer = in->size / last;
    for (int i = 0; i < outer; i++) {
        const float *x = in->data + i * last;
        float *y = out->data + i * last;
        /* Mean */
        float mean = 0.0f;
        for (int j = 0; j < last; j++) mean += x[j];
        mean /= (float)last;
        /* Variance */
        float var = 0.0f;
        for (int j = 0; j < last; j++) {
            float d = x[j] - mean;
            var += d * d;
        }
        var /= (float)last;
        float inv_std = 1.0f / sqrtf(var + eps);
        /* Normalize + affine */
        for (int j = 0; j < last; j++) {
            y[j] = (x[j] - mean) * inv_std;
            if (gamma) y[j] *= gamma->data[j];
            if (beta)  y[j] += beta->data[j];
        }
    }
}

float tensor_sum(const Tensor *t)
{
    float s = 0.0f;
    for (int i = 0; i < t->size; i++) s += t->data[i];
    return s;
}

float tensor_mean(const Tensor *t)
{
    return tensor_sum(t) / (float)t->size;
}

float tensor_l2_norm(const Tensor *t)
{
    float s = 0.0f;
    for (int i = 0; i < t->size; i++) s += t->data[i] * t->data[i];
    return sqrtf(s);
}

/* ═══════════════════════════════════════════════════════════════════
 * Misc
 * ═══════════════════════════════════════════════════════════════════ */

void tensor_concat(Tensor *dst, const Tensor *a, const Tensor *b, int axis)
{
    (void)axis; /* simplified: concat on last dim, 2D */
    int rows = a->shape[0];
    int ca = a->shape[1], cb = b->shape[1];
    for (int r = 0; r < rows; r++) {
        memcpy(dst->data + r * (ca + cb),
               a->data + r * ca, sizeof(float) * (size_t)ca);
        memcpy(dst->data + r * (ca + cb) + ca,
               b->data + r * cb, sizeof(float) * (size_t)cb);
    }
}

void tensor_copy(Tensor *dst, const Tensor *src)
{
    memcpy(dst->data, src->data, sizeof(float) * (size_t)src->size);
}

void tensor_print(const Tensor *t, const char *name)
{
    printf("Tensor '%s': shape=[", name);
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->shape[i], i < t->ndim - 1 ? ", " : "");
    }
    printf("], size=%d\n", t->size);
    int show = t->size < 20 ? t->size : 20;
    printf("  data: [");
    for (int i = 0; i < show; i++) {
        printf("%.4f%s", t->data[i], i < show - 1 ? ", " : "");
    }
    if (t->size > 20) printf(", ...");
    printf("]\n");
}
