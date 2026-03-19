/*
 * optimizer.c — Adam optimizer with learning rate scheduling
 * Voice Cloner TTS System
 */
#include "optimizer.h"
#include "../utils/memory.h"
#include "../utils/math_utils.h"
#include <math.h>
#include <string.h>

AdamOptimizer optimizer_create(float lr, float beta1, float beta2, float eps)
{
    AdamOptimizer opt;
    memset(&opt, 0, sizeof(AdamOptimizer));
    opt.lr    = lr;
    opt.beta1 = beta1;
    opt.beta2 = beta2;
    opt.eps   = eps;
    opt.max_grad_norm = 1.0f;
    opt.step  = 0;
    opt.n_params = 0;
    opt.warmup_steps = 4000;
    opt.base_lr = lr;
    return opt;
}

void optimizer_add_param(AdamOptimizer *opt, Tensor *param, Tensor *grad)
{
    if (opt->n_params >= MAX_PARAMS) {
        return; /* silently skip if we run out of param slots */
    }
    int i = opt->n_params;
    opt->params[i] = param;
    opt->grads[i]  = grad;
    opt->m[i] = tensor_zeros(param->ndim, param->shape);
    opt->v[i] = tensor_zeros(param->ndim, param->shape);
    opt->n_params++;
}

void optimizer_clip_grads(AdamOptimizer *opt, float max_norm)
{
    /* Compute global gradient norm */
    float total_norm = 0.0f;
    for (int i = 0; i < opt->n_params; i++) {
        for (int j = 0; j < opt->grads[i]->size; j++) {
            float g = opt->grads[i]->data[j];
            total_norm += g * g;
        }
    }
    total_norm = sqrtf(total_norm);

    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-10f);
        for (int i = 0; i < opt->n_params; i++) {
            tensor_scale(opt->grads[i], scale);
        }
    }
}

float optimizer_get_lr(AdamOptimizer *opt, int total_steps)
{
    int s = opt->step + 1;
    float lr;

    if (s < opt->warmup_steps) {
        /* Linear warmup */
        lr = opt->base_lr * (float)s / (float)opt->warmup_steps;
    } else {
        /* Cosine decay */
        float progress = (float)(s - opt->warmup_steps) /
                         (float)(total_steps - opt->warmup_steps + 1);
        if (progress > 1.0f) progress = 1.0f;
        lr = opt->base_lr * 0.5f * (1.0f + cosf((float)M_PI * progress));
    }

    return lr;
}

void optimizer_step(AdamOptimizer *opt)
{
    opt->step++;
    float lr = opt->lr;

    /* Bias correction */
    float bc1 = 1.0f - powf(opt->beta1, (float)opt->step);
    float bc2 = 1.0f - powf(opt->beta2, (float)opt->step);

    for (int i = 0; i < opt->n_params; i++) {
        int size = opt->params[i]->size;
        float *p = opt->params[i]->data;
        float *g = opt->grads[i]->data;
        float *m = opt->m[i].data;
        float *v = opt->v[i].data;

        for (int j = 0; j < size; j++) {
            /* Update moments */
            m[j] = opt->beta1 * m[j] + (1.0f - opt->beta1) * g[j];
            v[j] = opt->beta2 * v[j] + (1.0f - opt->beta2) * g[j] * g[j];

            /* Bias-corrected moments */
            float m_hat = m[j] / bc1;
            float v_hat = v[j] / bc2;

            /* Parameter update */
            p[j] -= lr * m_hat / (sqrtf(v_hat) + opt->eps);
        }
    }
}

void optimizer_zero_grad(AdamOptimizer *opt)
{
    for (int i = 0; i < opt->n_params; i++) {
        memset(opt->grads[i]->data, 0,
               sizeof(float) * (size_t)opt->grads[i]->size);
    }
}

void optimizer_free(AdamOptimizer *opt)
{
    for (int i = 0; i < opt->n_params; i++) {
        tensor_free(&opt->m[i]);
        tensor_free(&opt->v[i]);
    }
    opt->n_params = 0;
}
