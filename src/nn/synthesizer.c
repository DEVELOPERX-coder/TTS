/*
 * synthesizer.c — Tacotron-lite TTS Synthesizer
 * Voice Cloner TTS System
 */
#include "synthesizer.h"
#include "encoder.h"
#include "../utils/math_utils.h"
#include "../utils/memory.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════
 * Create
 * ═══════════════════════════════════════════════════════════════════ */

TTSSynthesizer synthesizer_create(int vocab_size)
{
    TTSSynthesizer s;
    memset(&s, 0, sizeof(TTSSynthesizer));
    s.vocab_size = vocab_size;

    /* Text encoder */
    s.text_embed = embedding_create(vocab_size, SYN_EMBED_DIM);
    for (int i = 0; i < 3; i++) {
        int in_ch = (i == 0) ? SYN_EMBED_DIM : SYN_ENC_DIM;
        s.enc_conv[i] = conv1d_create(in_ch, SYN_ENC_DIM, 5);
    }
    s.enc_gru = gru_stack_create(1, SYN_ENC_DIM, SYN_ENC_DIM / 2);

    /* Decoder */
    s.prenet = prenet_create(SYN_N_MELS, SYN_PRENET_DIM, 0.5f);

    /* Attention GRU input: prenet_dim + enc_dim (context) */
    s.attn_gru = gru_create(SYN_PRENET_DIM + SYN_ENC_DIM, SYN_DEC_DIM);

    /* Attention */
    s.attention = attention_create(SYN_ENC_DIM, SYN_DEC_DIM, SYN_ATTN_DIM);

    /* Decoder GRU input: attn_gru_output + context */
    s.dec_gru = gru_create(SYN_DEC_DIM + SYN_ENC_DIM, SYN_DEC_DIM);

    /* Output projections */
    s.mel_proj  = linear_create(SYN_DEC_DIM + SYN_ENC_DIM, SYN_N_MELS);
    s.stop_proj = linear_create(SYN_DEC_DIM + SYN_ENC_DIM, 1);
    s.postnet   = postnet_create(SYN_N_MELS);

    /* Speaker projection */
    s.speaker_proj = linear_create(SPEAKER_EMBED_DIM, SYN_ENC_DIM);

    return s;
}

/* ═══════════════════════════════════════════════════════════════════
 * Text Encoder
 * ═══════════════════════════════════════════════════════════════════ */

static Tensor encode_text(TTSSynthesizer *syn, const int *tokens, int n_tokens,
                          const Tensor *speaker_embed)
{
    /* 1. Embedding: [n_tokens, embed_dim] */
    int emb_shape[2] = { n_tokens, SYN_EMBED_DIM };
    Tensor emb = tensor_zeros(2, emb_shape);
    embedding_forward(&syn->text_embed, tokens, n_tokens, &emb);

    /* 2. Transpose for Conv1D: [embed_dim, n_tokens] */
    int conv_in_shape[2] = { SYN_EMBED_DIM, n_tokens };
    Tensor conv_in = tensor_zeros(2, conv_in_shape);
    for (int t = 0; t < n_tokens; t++) {
        for (int d = 0; d < SYN_EMBED_DIM; d++) {
            conv_in.data[d * n_tokens + t] = emb.data[t * SYN_EMBED_DIM + d];
        }
    }
    tensor_free(&emb);

    /* 3. 3x Conv1D + ReLU */
    Tensor current = conv_in;
    for (int i = 0; i < 3; i++) {
        int out_ch = syn->enc_conv[i].out_ch;
        int out_shape[2] = { out_ch, n_tokens };
        Tensor conv_out = tensor_zeros(2, out_shape);
        conv1d_forward(&syn->enc_conv[i], &current, &conv_out);

        /* ReLU */
        for (int j = 0; j < conv_out.size; j++) {
            conv_out.data[j] = relu_f(conv_out.data[j]);
        }

        tensor_free(&current);
        current = conv_out;
    }

    /* 4. Transpose back for GRU: [n_tokens, enc_dim] */
    int gru_in_shape[2] = { n_tokens, SYN_ENC_DIM };
    Tensor gru_in = tensor_zeros(2, gru_in_shape);
    for (int t = 0; t < n_tokens; t++) {
        for (int d = 0; d < SYN_ENC_DIM; d++) {
            gru_in.data[t * SYN_ENC_DIM + d] = current.data[d * n_tokens + t];
        }
    }
    tensor_free(&current);

    /* 5. BiGRU (simplified: forward GRU only, doubled hidden) */
    int gru_out_shape[2] = { n_tokens, SYN_ENC_DIM / 2 };
    Tensor gru_out = tensor_zeros(2, gru_out_shape);
    gru_stack_forward(&syn->enc_gru, &gru_in, &gru_out, NULL);
    tensor_free(&gru_in);

    /* Pad to full enc_dim by duplicating (simulating BiGRU) */
    int enc_out_shape[2] = { n_tokens, SYN_ENC_DIM };
    Tensor enc_output = tensor_zeros(2, enc_out_shape);
    for (int t = 0; t < n_tokens; t++) {
        for (int d = 0; d < SYN_ENC_DIM / 2; d++) {
            enc_output.data[t * SYN_ENC_DIM + d] = gru_out.data[t * (SYN_ENC_DIM/2) + d];
            enc_output.data[t * SYN_ENC_DIM + SYN_ENC_DIM/2 + d] =
                gru_out.data[t * (SYN_ENC_DIM/2) + d];
        }
    }
    tensor_free(&gru_out);

    /* 6. Add speaker embedding (broadcast across time) */
    int spk_shape[1] = { SYN_ENC_DIM };
    Tensor spk_proj = tensor_zeros(1, spk_shape);
    linear_forward(&syn->speaker_proj, speaker_embed, &spk_proj);

    for (int t = 0; t < n_tokens; t++) {
        for (int d = 0; d < SYN_ENC_DIM; d++) {
            enc_output.data[t * SYN_ENC_DIM + d] += spk_proj.data[d];
        }
    }
    tensor_free(&spk_proj);

    return enc_output;
}

/* ═══════════════════════════════════════════════════════════════════
 * Inference (autoregressive decoding)
 * ═══════════════════════════════════════════════════════════════════ */

int synthesizer_inference(TTSSynthesizer *syn,
                          const int *tokens, int n_tokens,
                          const Tensor *speaker_embed,
                          Tensor *mel_out)
{
    /* Encode text */
    Tensor memory = encode_text(syn, tokens, n_tokens, speaker_embed);

    /* Initialize decoder states */
    int dec_shape[1] = { SYN_DEC_DIM };
    Tensor h_attn = tensor_zeros(1, dec_shape);
    Tensor h_dec  = tensor_zeros(1, dec_shape);

    int ctx_shape[1] = { SYN_ENC_DIM };
    Tensor context = tensor_zeros(1, ctx_shape);

    int attn_shape[1] = { n_tokens };
    Tensor cum_attn = tensor_zeros(1, attn_shape);
    Tensor attn_weights = tensor_zeros(1, attn_shape);

    /* Initial mel frame (zeros) */
    int mel_frame_shape[1] = { SYN_N_MELS };
    Tensor mel_frame = tensor_zeros(1, mel_frame_shape);

    /* Output buffer */
    float *mel_buffer = (float *)safe_calloc(
        (size_t)SYN_N_MELS * (size_t)SYN_MAX_FRAMES, sizeof(float));

    int n_frames = 0;
    int prenet_shape[1] = { SYN_PRENET_DIM };
    int concat1_shape[1] = { SYN_PRENET_DIM + SYN_ENC_DIM };
    int concat2_shape[1] = { SYN_DEC_DIM + SYN_ENC_DIM };
    int stop_shape[1] = { 1 };

    for (int t = 0; t < SYN_MAX_FRAMES; t++) {
        /* 1. Prenet */
        Tensor prenet_out = tensor_zeros(1, prenet_shape);
        prenet_forward(&syn->prenet, &mel_frame, &prenet_out, 0);

        /* 2. Concat prenet + context → attention GRU input */
        Tensor attn_input = tensor_zeros(1, concat1_shape);
        memcpy(attn_input.data, prenet_out.data,
               sizeof(float) * (size_t)SYN_PRENET_DIM);
        memcpy(attn_input.data + SYN_PRENET_DIM, context.data,
               sizeof(float) * (size_t)SYN_ENC_DIM);

        /* 3. Attention GRU */
        Tensor h_attn_new = tensor_zeros(1, dec_shape);
        gru_forward(&syn->attn_gru, &attn_input, &h_attn, &h_attn_new);
        tensor_copy(&h_attn, &h_attn_new);

        /* 4. Attention */
        attention_forward(&syn->attention, &h_attn, &memory, &cum_attn,
                          &attn_weights, &context);

        /* Update cumulative attention */
        for (int i = 0; i < n_tokens; i++) {
            cum_attn.data[i] += attn_weights.data[i];
        }

        /* 5. Concat attn_gru output + context → decoder GRU input */
        Tensor dec_input = tensor_zeros(1, concat2_shape);
        memcpy(dec_input.data, h_attn.data,
               sizeof(float) * (size_t)SYN_DEC_DIM);
        memcpy(dec_input.data + SYN_DEC_DIM, context.data,
               sizeof(float) * (size_t)SYN_ENC_DIM);

        /* 6. Decoder GRU */
        Tensor h_dec_new = tensor_zeros(1, dec_shape);
        gru_forward(&syn->dec_gru, &dec_input, &h_dec, &h_dec_new);
        tensor_copy(&h_dec, &h_dec_new);

        /* 7. Concat decoder output + context → mel projection */
        Tensor proj_input = tensor_zeros(1, concat2_shape);
        memcpy(proj_input.data, h_dec.data,
               sizeof(float) * (size_t)SYN_DEC_DIM);
        memcpy(proj_input.data + SYN_DEC_DIM, context.data,
               sizeof(float) * (size_t)SYN_ENC_DIM);

        Tensor mel_pred = tensor_zeros(1, mel_frame_shape);
        linear_forward(&syn->mel_proj, &proj_input, &mel_pred);

        /* 8. Stop token prediction */
        Tensor stop_pred = tensor_zeros(1, stop_shape);
        linear_forward(&syn->stop_proj, &proj_input, &stop_pred);
        float stop_prob = sigmoid_f(stop_pred.data[0]);

        /* Store mel frame: layout [n_mels, n_frames] */
        for (int m = 0; m < SYN_N_MELS; m++) {
            mel_buffer[m * SYN_MAX_FRAMES + t] = mel_pred.data[m];
        }

        /* Update mel_frame for next step */
        tensor_copy(&mel_frame, &mel_pred);
        n_frames++;

        /* Cleanup per-step temporaries */
        tensor_free(&prenet_out);
        tensor_free(&attn_input);
        tensor_free(&h_attn_new);
        tensor_free(&dec_input);
        tensor_free(&h_dec_new);
        tensor_free(&proj_input);
        tensor_free(&mel_pred);
        tensor_free(&stop_pred);

        /* Check stop condition */
        if (stop_prob > 0.5f && t > 10) break;
    }

    /* Build output mel tensor: [n_mels, n_frames] */
    int out_shape[2] = { SYN_N_MELS, n_frames };
    *mel_out = tensor_zeros(2, out_shape);
    for (int m = 0; m < SYN_N_MELS; m++) {
        memcpy(mel_out->data + m * n_frames,
               mel_buffer + m * SYN_MAX_FRAMES,
               sizeof(float) * (size_t)n_frames);
    }

    /* Apply postnet */
    Tensor mel_post = tensor_zeros(2, out_shape);
    postnet_forward(&syn->postnet, mel_out, &mel_post);
    tensor_copy(mel_out, &mel_post);

    /* Cleanup */
    free(mel_buffer);
    tensor_free(&memory);
    tensor_free(&h_attn);
    tensor_free(&h_dec);
    tensor_free(&context);
    tensor_free(&cum_attn);
    tensor_free(&attn_weights);
    tensor_free(&mel_frame);
    tensor_free(&mel_post);

    return n_frames;
}

/* ═══════════════════════════════════════════════════════════════════
 * Training step (teacher forcing)
 * ═══════════════════════════════════════════════════════════════════ */

float synthesizer_train_step(TTSSynthesizer *syn,
                             const int *tokens, int n_tokens,
                             const Tensor *speaker_embed,
                             const Tensor *target_mel)
{
    int n_frames = target_mel->shape[1];

    /* Encode text */
    Tensor memory = encode_text(syn, tokens, n_tokens, speaker_embed);

    /* Initialize decoder states */
    int dec_shape[1] = { SYN_DEC_DIM };
    Tensor h_attn = tensor_zeros(1, dec_shape);
    Tensor h_dec  = tensor_zeros(1, dec_shape);

    int ctx_shape[1] = { SYN_ENC_DIM };
    Tensor context = tensor_zeros(1, ctx_shape);

    int attn_shape[1] = { n_tokens };
    Tensor cum_attn = tensor_zeros(1, attn_shape);
    Tensor attn_weights = tensor_zeros(1, attn_shape);

    int mel_f_shape[1] = { SYN_N_MELS };
    Tensor mel_frame = tensor_zeros(1, mel_f_shape);

    float total_loss = 0.0f;

    int prenet_shape[1] = { SYN_PRENET_DIM };
    int concat1_shape[1] = { SYN_PRENET_DIM + SYN_ENC_DIM };
    int concat2_shape[1] = { SYN_DEC_DIM + SYN_ENC_DIM };
    int stop_shape[1] = { 1 };

    for (int t = 0; t < n_frames; t++) {
        /* Use teacher forcing: use target mel from previous frame */
        if (t > 0) {
            for (int m = 0; m < SYN_N_MELS; m++) {
                mel_frame.data[m] = target_mel->data[m * n_frames + (t - 1)];
            }
        }

        /* Forward pass (same as inference) */
        Tensor prenet_out = tensor_zeros(1, prenet_shape);
        prenet_forward(&syn->prenet, &mel_frame, &prenet_out, 1);

        Tensor attn_input = tensor_zeros(1, concat1_shape);
        memcpy(attn_input.data, prenet_out.data,
               sizeof(float) * (size_t)SYN_PRENET_DIM);
        memcpy(attn_input.data + SYN_PRENET_DIM, context.data,
               sizeof(float) * (size_t)SYN_ENC_DIM);

        Tensor h_attn_new = tensor_zeros(1, dec_shape);
        gru_forward(&syn->attn_gru, &attn_input, &h_attn, &h_attn_new);
        tensor_copy(&h_attn, &h_attn_new);

        attention_forward(&syn->attention, &h_attn, &memory, &cum_attn,
                          &attn_weights, &context);
        for (int i = 0; i < n_tokens; i++) {
            cum_attn.data[i] += attn_weights.data[i];
        }

        Tensor dec_input = tensor_zeros(1, concat2_shape);
        memcpy(dec_input.data, h_attn.data,
               sizeof(float) * (size_t)SYN_DEC_DIM);
        memcpy(dec_input.data + SYN_DEC_DIM, context.data,
               sizeof(float) * (size_t)SYN_ENC_DIM);

        Tensor h_dec_new = tensor_zeros(1, dec_shape);
        gru_forward(&syn->dec_gru, &dec_input, &h_dec, &h_dec_new);
        tensor_copy(&h_dec, &h_dec_new);

        Tensor proj_input = tensor_zeros(1, concat2_shape);
        memcpy(proj_input.data, h_dec.data,
               sizeof(float) * (size_t)SYN_DEC_DIM);
        memcpy(proj_input.data + SYN_DEC_DIM, context.data,
               sizeof(float) * (size_t)SYN_ENC_DIM);

        Tensor mel_pred = tensor_zeros(1, mel_f_shape);
        linear_forward(&syn->mel_proj, &proj_input, &mel_pred);

        /* Compute MSE loss for this frame */
        float frame_loss = 0.0f;
        for (int m = 0; m < SYN_N_MELS; m++) {
            float diff = mel_pred.data[m] - target_mel->data[m * n_frames + t];
            frame_loss += diff * diff;
        }
        total_loss += frame_loss / (float)SYN_N_MELS;

        /* Cleanup temporaries */
        tensor_free(&prenet_out);
        tensor_free(&attn_input);
        tensor_free(&h_attn_new);
        tensor_free(&dec_input);
        tensor_free(&h_dec_new);
        tensor_free(&proj_input);
        tensor_free(&mel_pred);
    }

    /* Cleanup */
    tensor_free(&memory);
    tensor_free(&h_attn);
    tensor_free(&h_dec);
    tensor_free(&context);
    tensor_free(&cum_attn);
    tensor_free(&attn_weights);
    tensor_free(&mel_frame);

    return total_loss / (float)n_frames;
}

/* ═══════════════════════════════════════════════════════════════════
 * Save / Load
 * ═══════════════════════════════════════════════════════════════════ */

static void save_tensor_f(FILE *f, const Tensor *t)
{
    fwrite(&t->ndim, sizeof(int), 1, f);
    fwrite(t->shape, sizeof(int), (size_t)t->ndim, f);
    fwrite(&t->size, sizeof(int), 1, f);
    fwrite(t->data, sizeof(float), (size_t)t->size, f);
}

static void load_tensor_f(FILE *f, Tensor *t)
{
    int ndim;
    fread(&ndim, sizeof(int), 1, f);
    int shape[TENSOR_MAX_DIMS];
    fread(shape, sizeof(int), (size_t)ndim, f);
    int size;
    fread(&size, sizeof(int), 1, f);
    if (t->data) tensor_free(t);
    *t = tensor_create(ndim, shape);
    fread(t->data, sizeof(float), (size_t)size, f);
}

static void save_linear_f(FILE *f, const LinearLayer *l)
{
    save_tensor_f(f, &l->W);
    save_tensor_f(f, &l->b);
}

static void load_linear_f(FILE *f, LinearLayer *l)
{
    load_tensor_f(f, &l->W);
    load_tensor_f(f, &l->b);
}

int synthesizer_save(const TTSSynthesizer *syn, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int magic = 0x54545353; /* "TTSS" */
    int version = 1;
    fwrite(&magic, sizeof(int), 1, f);
    fwrite(&version, sizeof(int), 1, f);

    /* Text embed */
    save_tensor_f(f, &syn->text_embed.W);

    /* Encoder convs */
    for (int i = 0; i < 3; i++) {
        save_tensor_f(f, &syn->enc_conv[i].W);
        save_tensor_f(f, &syn->enc_conv[i].b);
    }

    /* Key linear layers */
    save_linear_f(f, &syn->mel_proj);
    save_linear_f(f, &syn->stop_proj);
    save_linear_f(f, &syn->speaker_proj);

    fclose(f);
    printf("[INFO] Synthesizer saved to %s\n", path);
    return 0;
}

int synthesizer_load(TTSSynthesizer *syn, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    int magic, version;
    fread(&magic, sizeof(int), 1, f);
    fread(&version, sizeof(int), 1, f);

    if (magic != 0x54545353) {
        fclose(f);
        return -1;
    }

    load_tensor_f(f, &syn->text_embed.W);

    for (int i = 0; i < 3; i++) {
        load_tensor_f(f, &syn->enc_conv[i].W);
        load_tensor_f(f, &syn->enc_conv[i].b);
    }

    load_linear_f(f, &syn->mel_proj);
    load_linear_f(f, &syn->stop_proj);
    load_linear_f(f, &syn->speaker_proj);

    fclose(f);
    printf("[INFO] Synthesizer loaded from %s\n", path);
    return 0;
}

void synthesizer_free(TTSSynthesizer *syn)
{
    embedding_free(&syn->text_embed);
    for (int i = 0; i < 3; i++) conv1d_free(&syn->enc_conv[i]);
    gru_stack_free(&syn->enc_gru);
    prenet_free(&syn->prenet);
    gru_free(&syn->attn_gru);
    gru_free(&syn->dec_gru);
    attention_free(&syn->attention);
    linear_free(&syn->mel_proj);
    linear_free(&syn->stop_proj);
    postnet_free(&syn->postnet);
    linear_free(&syn->speaker_proj);
}
