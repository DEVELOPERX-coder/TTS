/*
 * layers.c — Neural network layers implementation
 * Voice Cloner TTS System
 */
#include "layers.h"
#include "../utils/math_utils.h"
#include "../utils/memory.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════
 * He initialization
 * ═══════════════════════════════════════════════════════════════════ */

void he_init(Tensor *W, int fan_in)
{
    float std = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < W->size; i++) {
        W->data[i] = randn() * std;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Linear layer
 * ═══════════════════════════════════════════════════════════════════ */

LinearLayer linear_create(int in_dim, int out_dim)
{
    LinearLayer l;
    l.in_dim  = in_dim;
    l.out_dim = out_dim;

    int w_shape[2] = { out_dim, in_dim };
    l.W  = tensor_create(2, w_shape);
    l.dW = tensor_zeros(2, w_shape);
    he_init(&l.W, in_dim);

    int b_shape[1] = { out_dim };
    l.b  = tensor_zeros(1, b_shape);
    l.db = tensor_zeros(1, b_shape);

    return l;
}

void linear_forward(LinearLayer *l, const Tensor *input, Tensor *output)
{
    /* input: [batch, in_dim] or [in_dim]
     * output: [batch, out_dim] or [out_dim]
     * y = x @ W^T + b */
    int batch = 1;
    if (input->ndim == 2) batch = input->shape[0];

    for (int b = 0; b < batch; b++) {
        const float *x = input->data + b * l->in_dim;
        float *y = output->data + b * l->out_dim;
        for (int o = 0; o < l->out_dim; o++) {
            float sum = l->b.data[o];
            const float *w_row = l->W.data + o * l->in_dim;
            for (int i = 0; i < l->in_dim; i++) {
                sum += w_row[i] * x[i];
            }
            y[o] = sum;
        }
    }
}

void linear_free(LinearLayer *l)
{
    tensor_free(&l->W);
    tensor_free(&l->b);
    tensor_free(&l->dW);
    tensor_free(&l->db);
}

/* ═══════════════════════════════════════════════════════════════════
 * GRU cell
 * ═══════════════════════════════════════════════════════════════════ */

GRUCell gru_create(int input_dim, int hidden_dim)
{
    GRUCell g;
    g.input_dim  = input_dim;
    g.hidden_dim = hidden_dim;
    g.W_z = linear_create(input_dim, hidden_dim);
    g.W_r = linear_create(input_dim, hidden_dim);
    g.W_h = linear_create(input_dim, hidden_dim);
    g.U_z = linear_create(hidden_dim, hidden_dim);
    g.U_r = linear_create(hidden_dim, hidden_dim);
    g.U_h = linear_create(hidden_dim, hidden_dim);
    return g;
}

void gru_forward(GRUCell *g, const Tensor *x, const Tensor *h_prev, Tensor *h_next)
{
    int hd = g->hidden_dim;

    /* Temp tensors */
    int h_shape[1] = { hd };
    Tensor wz_out = tensor_zeros(1, h_shape);
    Tensor uz_out = tensor_zeros(1, h_shape);
    Tensor wr_out = tensor_zeros(1, h_shape);
    Tensor ur_out = tensor_zeros(1, h_shape);
    Tensor wh_out = tensor_zeros(1, h_shape);
    Tensor uh_out = tensor_zeros(1, h_shape);

    /* z = σ(W_z·x + U_z·h) */
    linear_forward(&g->W_z, x, &wz_out);
    linear_forward(&g->U_z, h_prev, &uz_out);

    /* r = σ(W_r·x + U_r·h) */
    linear_forward(&g->W_r, x, &wr_out);
    linear_forward(&g->U_r, h_prev, &ur_out);

    float *z = (float *)safe_malloc(sizeof(float) * (size_t)hd);
    float *r = (float *)safe_malloc(sizeof(float) * (size_t)hd);

    for (int i = 0; i < hd; i++) {
        z[i] = sigmoid_f(wz_out.data[i] + uz_out.data[i]);
        r[i] = sigmoid_f(wr_out.data[i] + ur_out.data[i]);
    }

    /* h_tilde = tanh(W_h·x + U_h·(r⊙h)) */
    linear_forward(&g->W_h, x, &wh_out);

    /* Compute r⊙h */
    Tensor rh = tensor_clone(h_prev);
    for (int i = 0; i < hd; i++) rh.data[i] *= r[i];
    linear_forward(&g->U_h, &rh, &uh_out);

    /* h_next = (1-z)⊙h + z⊙h_tilde */
    for (int i = 0; i < hd; i++) {
        float h_tilde = tanh_f(wh_out.data[i] + uh_out.data[i]);
        h_next->data[i] = (1.0f - z[i]) * h_prev->data[i] + z[i] * h_tilde;
    }

    free(z);
    free(r);
    tensor_free(&wz_out);
    tensor_free(&uz_out);
    tensor_free(&wr_out);
    tensor_free(&ur_out);
    tensor_free(&wh_out);
    tensor_free(&uh_out);
    tensor_free(&rh);
}

void gru_free(GRUCell *g)
{
    linear_free(&g->W_z);
    linear_free(&g->W_r);
    linear_free(&g->W_h);
    linear_free(&g->U_z);
    linear_free(&g->U_r);
    linear_free(&g->U_h);
}

/* ═══════════════════════════════════════════════════════════════════
 * Multi-layer GRU stack
 * ═══════════════════════════════════════════════════════════════════ */

GRUStack gru_stack_create(int n_layers, int input_dim, int hidden_dim)
{
    GRUStack s;
    s.n_layers   = n_layers;
    s.input_dim  = input_dim;
    s.hidden_dim = hidden_dim;
    s.cells = (GRUCell *)safe_malloc(sizeof(GRUCell) * (size_t)n_layers);

    s.cells[0] = gru_create(input_dim, hidden_dim);
    for (int i = 1; i < n_layers; i++) {
        s.cells[i] = gru_create(hidden_dim, hidden_dim);
    }
    return s;
}

void gru_stack_forward(GRUStack *g, const Tensor *input, Tensor *output, Tensor *final_h)
{
    int seq_len = input->shape[0];
    int hd = g->hidden_dim;

    /* Allocate hidden states for each layer */
    int h_shape[1] = { hd };
    Tensor *h_states = (Tensor *)safe_malloc(sizeof(Tensor) * (size_t)g->n_layers);
    for (int l = 0; l < g->n_layers; l++) {
        h_states[l] = tensor_zeros(1, h_shape);
    }

    /* Process each time step */
    int in_dim = input->shape[1];
    int x_shape[1] = { in_dim };
    int xh_shape[1] = { hd };

    for (int t = 0; t < seq_len; t++) {
        /* Input for first layer */
        Tensor xt;
        xt.data = input->data + t * in_dim;
        xt.ndim = 1;
        xt.shape[0] = in_dim;
        xt.size = in_dim;
        xt.owns_data = 0;

        Tensor h_new = tensor_zeros(1, h_shape);

        for (int l = 0; l < g->n_layers; l++) {
            const Tensor *layer_input = (l == 0) ? &xt : &h_states[l - 1];

            /* If not first layer, we need to use the just-updated h from previous layer */
            if (l > 0) {
                /* layer_input is h_states[l-1] which was just updated */
            }

            gru_forward(&g->cells[l], layer_input, &h_states[l], &h_new);
            tensor_copy(&h_states[l], &h_new);
        }

        /* Copy last layer's output to output tensor */
        if (output) {
            memcpy(output->data + t * hd, h_states[g->n_layers - 1].data,
                   sizeof(float) * (size_t)hd);
        }
        tensor_free(&h_new);
    }

    /* Copy final hidden state */
    if (final_h) {
        tensor_copy(final_h, &h_states[g->n_layers - 1]);
    }

    for (int l = 0; l < g->n_layers; l++) {
        tensor_free(&h_states[l]);
    }
    free(h_states);
}

void gru_stack_free(GRUStack *g)
{
    for (int i = 0; i < g->n_layers; i++) {
        gru_free(&g->cells[i]);
    }
    free(g->cells);
}

/* ═══════════════════════════════════════════════════════════════════
 * Conv1D layer
 * ═══════════════════════════════════════════════════════════════════ */

Conv1DLayer conv1d_create(int in_ch, int out_ch, int kernel)
{
    Conv1DLayer l;
    l.in_ch   = in_ch;
    l.out_ch  = out_ch;
    l.kernel  = kernel;
    l.padding = kernel / 2; /* same padding */

    int w_shape[3] = { out_ch, in_ch, kernel };
    l.W  = tensor_create(3, w_shape);
    l.dW = tensor_zeros(3, w_shape);
    he_init(&l.W, in_ch * kernel);

    int b_shape[1] = { out_ch };
    l.b  = tensor_zeros(1, b_shape);
    l.db = tensor_zeros(1, b_shape);

    return l;
}

void conv1d_forward(Conv1DLayer *l, const Tensor *input, Tensor *output)
{
    /* input: [in_ch, seq_len], output: [out_ch, seq_len] */
    int seq = input->shape[1];
    int pad = l->padding;

    for (int oc = 0; oc < l->out_ch; oc++) {
        for (int t = 0; t < seq; t++) {
            float sum = l->b.data[oc];
            for (int ic = 0; ic < l->in_ch; ic++) {
                for (int k = 0; k < l->kernel; k++) {
                    int pos = t + k - pad;
                    if (pos >= 0 && pos < seq) {
                        sum += l->W.data[(oc * l->in_ch + ic) * l->kernel + k]
                             * input->data[ic * seq + pos];
                    }
                }
            }
            output->data[oc * seq + t] = sum;
        }
    }
}

void conv1d_free(Conv1DLayer *l)
{
    tensor_free(&l->W);
    tensor_free(&l->b);
    tensor_free(&l->dW);
    tensor_free(&l->db);
}

/* ═══════════════════════════════════════════════════════════════════
 * Embedding layer
 * ═══════════════════════════════════════════════════════════════════ */

EmbeddingLayer embedding_create(int vocab_size, int embed_dim)
{
    EmbeddingLayer e;
    e.vocab_size = vocab_size;
    e.embed_dim  = embed_dim;

    int shape[2] = { vocab_size, embed_dim };
    e.W  = tensor_create(2, shape);
    e.dW = tensor_zeros(2, shape);

    /* Xavier-ish initialization */
    float std = 1.0f / sqrtf((float)embed_dim);
    for (int i = 0; i < e.W.size; i++) {
        e.W.data[i] = randn() * std;
    }

    return e;
}

void embedding_forward(EmbeddingLayer *e, const int *tokens, int n, Tensor *output)
{
    /* output: [n, embed_dim] */
    for (int i = 0; i < n; i++) {
        int tok = tokens[i];
        if (tok < 0 || tok >= e->vocab_size) tok = 0;
        memcpy(output->data + i * e->embed_dim,
               e->W.data + tok * e->embed_dim,
               sizeof(float) * (size_t)e->embed_dim);
    }
}

void embedding_free(EmbeddingLayer *e)
{
    tensor_free(&e->W);
    tensor_free(&e->dW);
}

/* ═══════════════════════════════════════════════════════════════════
 * LayerNorm
 * ═══════════════════════════════════════════════════════════════════ */

LayerNormLayer layer_norm_create(int dim)
{
    LayerNormLayer ln;
    ln.dim = dim;
    ln.eps = 1e-5f;

    int shape[1] = { dim };
    ln.gamma = tensor_ones(1, shape);
    ln.beta  = tensor_zeros(1, shape);

    return ln;
}

void layer_norm_forward(LayerNormLayer *ln, const Tensor *input, Tensor *output)
{
    tensor_layer_norm(output, input, &ln->gamma, &ln->beta, ln->eps);
}

void layer_norm_free(LayerNormLayer *ln)
{
    tensor_free(&ln->gamma);
    tensor_free(&ln->beta);
}

/* ═══════════════════════════════════════════════════════════════════
 * Prenet (2-layer FC + ReLU + Dropout)
 * ═══════════════════════════════════════════════════════════════════ */

PrenetLayer prenet_create(int input_dim, int hidden_dim, float dropout)
{
    PrenetLayer p;
    p.input_dim  = input_dim;
    p.hidden_dim = hidden_dim;
    p.dropout    = dropout;
    p.fc1 = linear_create(input_dim, hidden_dim);
    p.fc2 = linear_create(hidden_dim, hidden_dim);
    return p;
}

void prenet_forward(PrenetLayer *p, const Tensor *input, Tensor *output, int training)
{
    int batch = (input->ndim == 2) ? input->shape[0] : 1;
    int h = p->hidden_dim;

    /* Layer 1: FC + ReLU + Dropout */
    int mid_shape[2] = { batch, h };
    Tensor mid = tensor_zeros(2, mid_shape);
    linear_forward(&p->fc1, input, &mid);

    for (int i = 0; i < mid.size; i++) {
        mid.data[i] = relu_f(mid.data[i]);
        if (training && uniform(0.0f, 1.0f) < p->dropout) {
            mid.data[i] = 0.0f;
        }
    }

    /* Layer 2: FC + ReLU + Dropout */
    linear_forward(&p->fc2, &mid, output);
    for (int i = 0; i < output->size; i++) {
        output->data[i] = relu_f(output->data[i]);
        if (training && uniform(0.0f, 1.0f) < p->dropout) {
            output->data[i] = 0.0f;
        }
    }

    tensor_free(&mid);
}

void prenet_free(PrenetLayer *p)
{
    linear_free(&p->fc1);
    linear_free(&p->fc2);
}

/* ═══════════════════════════════════════════════════════════════════
 * PostNet (5x Conv1D + BatchNorm + Tanh)
 * ═══════════════════════════════════════════════════════════════════ */

PostnetLayer postnet_create(int n_mels)
{
    PostnetLayer p;
    p.n_mels = n_mels;

    /* First conv: n_mels → POSTNET_CH */
    p.convs[0] = conv1d_create(n_mels, POSTNET_CH, POSTNET_KERNEL);
    /* Middle convs: POSTNET_CH → POSTNET_CH */
    for (int i = 1; i < POSTNET_CONVS - 1; i++) {
        p.convs[i] = conv1d_create(POSTNET_CH, POSTNET_CH, POSTNET_KERNEL);
    }
    /* Last conv: POSTNET_CH → n_mels */
    p.convs[POSTNET_CONVS - 1] = conv1d_create(POSTNET_CH, n_mels, POSTNET_KERNEL);

    return p;
}

void postnet_forward(PostnetLayer *p, const Tensor *input, Tensor *output)
{
    int n_frames = input->shape[1];

    /* Clone input for residual */
    Tensor residual = tensor_clone(input);

    /* Process through conv layers */
    Tensor current = tensor_clone(input);

    for (int i = 0; i < POSTNET_CONVS; i++) {
        int out_ch = p->convs[i].out_ch;
        int conv_shape[2] = { out_ch, n_frames };
        Tensor conv_out = tensor_zeros(2, conv_shape);
        conv1d_forward(&p->convs[i], &current, &conv_out);

        /* Activation: tanh for all except last layer */
        if (i < POSTNET_CONVS - 1) {
            for (int j = 0; j < conv_out.size; j++) {
                conv_out.data[j] = tanh_f(conv_out.data[j]);
            }
        }

        tensor_free(&current);
        current = conv_out;
    }

    /* Residual connection: output = input + postnet(input) */
    tensor_add(output, &residual, &current);

    tensor_free(&current);
    tensor_free(&residual);
}

void postnet_free(PostnetLayer *p)
{
    for (int i = 0; i < POSTNET_CONVS; i++) {
        conv1d_free(&p->convs[i]);
    }
}
