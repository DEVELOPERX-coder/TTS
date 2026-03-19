/*
 * tensor.h — N-dimensional tensor operations
 * Voice Cloner TTS System
 */
#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

#define TENSOR_MAX_DIMS 4

typedef struct {
    float *data;
    int    shape[TENSOR_MAX_DIMS];
    int    ndim;
    int    size;      /* total number of elements */
    int    owns_data; /* 1 = we allocated data, 0 = view */
} Tensor;

/* ── Lifecycle ──────────────────────────────────────────────────── */
Tensor tensor_create(int ndim, const int *shape);
Tensor tensor_zeros(int ndim, const int *shape);
Tensor tensor_ones(int ndim, const int *shape);
Tensor tensor_randn(int ndim, const int *shape);
Tensor tensor_clone(const Tensor *t);
void   tensor_free(Tensor *t);

/* ── Views ──────────────────────────────────────────────────────── */
Tensor tensor_view(Tensor *t, int ndim, const int *shape);

/* ── Element-wise ops ───────────────────────────────────────────── */
void tensor_add(Tensor *dst, const Tensor *a, const Tensor *b);
void tensor_add_inplace(Tensor *dst, const Tensor *src);
void tensor_sub(Tensor *dst, const Tensor *a, const Tensor *b);
void tensor_mul_elem(Tensor *dst, const Tensor *a, const Tensor *b);
void tensor_scale(Tensor *t, float s);
void tensor_fill(Tensor *t, float val);

/* ── Matrix ops ─────────────────────────────────────────────────── */
/* C = A @ B  where A:[M,K], B:[K,N], C:[M,N] */
void tensor_matmul(Tensor *C, const Tensor *A, const Tensor *B);

/* C = A^T @ B */
void tensor_matmul_transA(Tensor *C, const Tensor *A, const Tensor *B);

/* C = A @ B^T */
void tensor_matmul_transB(Tensor *C, const Tensor *A, const Tensor *B);

/* ── Reductions ─────────────────────────────────────────────────── */
void tensor_softmax(Tensor *t, int axis);
void tensor_layer_norm(Tensor *out, const Tensor *in, const Tensor *gamma,
                       const Tensor *beta, float eps);
float tensor_sum(const Tensor *t);
float tensor_mean(const Tensor *t);
float tensor_l2_norm(const Tensor *t);

/* ── Misc ───────────────────────────────────────────────────────── */
void tensor_concat(Tensor *dst, const Tensor *a, const Tensor *b, int axis);
void tensor_copy(Tensor *dst, const Tensor *src);
void tensor_print(const Tensor *t, const char *name);

#endif /* TENSOR_H */
