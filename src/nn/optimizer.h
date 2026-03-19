/*
 * optimizer.h — Adam optimizer with learning rate scheduling
 * Voice Cloner TTS System
 */
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"

#define MAX_PARAMS 512

typedef struct {
    Tensor *params[MAX_PARAMS];  /* pointers to parameter tensors */
    Tensor *grads[MAX_PARAMS];   /* pointers to gradient tensors */
    Tensor  m[MAX_PARAMS];       /* first moment (Adam) */
    Tensor  v[MAX_PARAMS];       /* second moment (Adam) */
    int     n_params;
    float   lr;
    float   beta1, beta2, eps;
    float   max_grad_norm;
    int     step;
    /* LR schedule */
    int     warmup_steps;
    float   base_lr;
} AdamOptimizer;

AdamOptimizer optimizer_create(float lr, float beta1, float beta2, float eps);

/* Register a parameter tensor and its gradient for optimization. */
void optimizer_add_param(AdamOptimizer *opt, Tensor *param, Tensor *grad);

/* Perform one step of Adam optimization. */
void optimizer_step(AdamOptimizer *opt);

/* Zero all gradients. */
void optimizer_zero_grad(AdamOptimizer *opt);

/* Clip gradients by global norm. */
void optimizer_clip_grads(AdamOptimizer *opt, float max_norm);

/* Get current learning rate (with warmup + cosine decay). */
float optimizer_get_lr(AdamOptimizer *opt, int total_steps);

void optimizer_free(AdamOptimizer *opt);

#endif /* OPTIMIZER_H */
