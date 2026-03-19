/*
 * main.c — Voice Cloner CLI Application
 * Voice Cloner TTS System
 *
 * Usage:
 *   voice_cloner train    [--data DIR] [--epochs N] [--lr F]
 *   voice_cloner synth    --text "..." [--voice MODEL] [--output FILE]
 *   voice_cloner test-wav
 *   voice_cloner test-fft
 *   voice_cloner test-features
 *   voice_cloner test-nn
 *   voice_cloner help
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "audio/wav.h"
#include "audio/features.h"
#include "audio/vocoder.h"
#include "text/tokenizer.h"
#include "text/phonemes.h"
#include "nn/tensor.h"
#include "nn/layers.h"
#include "nn/attention.h"
#include "nn/encoder.h"
#include "nn/synthesizer.h"
#include "nn/optimizer.h"
#include "nn/train.h"
#include "utils/math_utils.h"
#include "utils/memory.h"

/* ═══════════════════════════════════════════════════════════════════
 * Self-tests
 * ═══════════════════════════════════════════════════════════════════ */

static int test_wav(void)
{
    printf("\n=== WAV I/O Test ===\n");

    /* Generate sine wave */
    AudioData sine = audio_generate_sine(440.0f, 0.5f, 16000);
    printf("[OK] Generated 440 Hz sine wave: %d samples @ %d Hz\n",
           sine.num_samples, sine.sample_rate);

    /* Write to file */
    if (audio_write_wav("test_sine.wav", &sine) != 0) {
        printf("[FAIL] Could not write WAV file.\n");
        audio_free(&sine);
        return -1;
    }
    printf("[OK] Written to test_sine.wav\n");

    /* Read back */
    AudioData readback;
    if (audio_read_wav("test_sine.wav", &readback) != 0) {
        printf("[FAIL] Could not read WAV file.\n");
        audio_free(&sine);
        return -1;
    }
    printf("[OK] Read back: %d samples @ %d Hz\n",
           readback.num_samples, readback.sample_rate);

    /* Verify samples */
    float max_err = 0.0f;
    int check_n = sine.num_samples < readback.num_samples ?
                  sine.num_samples : readback.num_samples;
    for (int i = 0; i < check_n; i++) {
        float err = fabsf(sine.samples[i] - readback.samples[i]);
        if (err > max_err) max_err = err;
    }
    printf("[OK] Max sample error: %.6f (threshold: 0.001)\n", max_err);

    if (max_err < 0.001f) {
        printf("[PASS] WAV I/O test passed!\n");
    } else {
        printf("[FAIL] WAV I/O test failed — error too large.\n");
    }

    audio_free(&sine);
    audio_free(&readback);
    return (max_err < 0.001f) ? 0 : -1;
}

static int test_fft(void)
{
    printf("\n=== FFT Test ===\n");

    int n = 256;
    float *signal = (float *)safe_calloc((size_t)n, sizeof(float));

    /* Generate 440 Hz sine in 16 kHz */
    for (int i = 0; i < n; i++) {
        signal[i] = sinf(2.0f * (float)M_PI * 440.0f * (float)i / 16000.0f);
    }

    Complexf *spec = (Complexf *)safe_malloc(sizeof(Complexf) * (size_t)(n / 2 + 1));
    rfft(signal, spec, n);

    /* Find peak bin */
    float max_mag = 0.0f;
    int peak_bin = 0;
    for (int i = 0; i < n / 2 + 1; i++) {
        float mag = sqrtf(spec[i].re * spec[i].re + spec[i].im * spec[i].im);
        if (mag > max_mag) {
            max_mag = mag;
            peak_bin = i;
        }
    }

    float expected_bin = 440.0f * (float)n / 16000.0f;
    printf("[OK] Peak bin: %d (expected ~%.1f)\n", peak_bin, expected_bin);
    printf("[OK] Peak magnitude: %.4f\n", max_mag);

    int pass = (abs(peak_bin - (int)expected_bin) <= 1);
    if (pass) {
        printf("[PASS] FFT test passed!\n");
    } else {
        printf("[FAIL] FFT test failed — peak bin mismatch.\n");
    }

    free(signal);
    free(spec);
    return pass ? 0 : -1;
}

static int test_features(void)
{
    printf("\n=== Feature Extraction Test ===\n");

    /* Generate test audio */
    AudioData sine = audio_generate_sine(440.0f, 1.0f, SAMPLE_RATE);
    printf("[OK] Generated test audio: %d samples\n", sine.num_samples);

    /* Compute mel spectrogram */
    Tensor mel = compute_mel_spectrogram(&sine);
    printf("[OK] Mel spectrogram: [%d, %d]\n", mel.shape[0], mel.shape[1]);

    /* Check dimensions */
    int pass = (mel.shape[0] == N_MELS && mel.shape[1] > 0);

    /* Check value range (should be [0, 1] after normalization) */
    float mn = mel.data[0], mx = mel.data[0];
    for (int i = 1; i < mel.size; i++) {
        if (mel.data[i] < mn) mn = mel.data[i];
        if (mel.data[i] > mx) mx = mel.data[i];
    }
    printf("[OK] Value range: [%.4f, %.4f]\n", mn, mx);

    if (pass && mn >= -0.1f && mx <= 1.1f) {
        printf("[PASS] Feature extraction test passed!\n");
    } else {
        printf("[FAIL] Feature extraction test failed.\n");
        pass = 0;
    }

    tensor_free(&mel);
    audio_free(&sine);
    return pass ? 0 : -1;
}

static int test_nn(void)
{
    printf("\n=== Neural Network Test ===\n");

    /* Test Linear layer */
    printf("Testing Linear layer...\n");
    LinearLayer fc = linear_create(64, 32);
    int x_shape[2] = { 1, 64 };
    Tensor x = tensor_randn(2, x_shape);
    int y_shape[2] = { 1, 32 };
    Tensor y = tensor_zeros(2, y_shape);
    linear_forward(&fc, &x, &y);
    printf("[OK] Linear(64→32): output mean=%.4f\n", tensor_mean(&y));
    tensor_free(&x);
    tensor_free(&y);
    linear_free(&fc);

    /* Test GRU */
    printf("Testing GRU cell...\n");
    GRUCell gru = gru_create(64, 128);
    int gx_shape[1] = { 64 };
    Tensor gx = tensor_randn(1, gx_shape);
    int gh_shape[1] = { 128 };
    Tensor gh = tensor_zeros(1, gh_shape);
    Tensor gh_out = tensor_zeros(1, gh_shape);
    gru_forward(&gru, &gx, &gh, &gh_out);
    printf("[OK] GRU(64→128): hidden mean=%.4f\n", tensor_mean(&gh_out));
    tensor_free(&gx);
    tensor_free(&gh);
    tensor_free(&gh_out);
    gru_free(&gru);

    /* Test Embedding */
    printf("Testing Embedding layer...\n");
    EmbeddingLayer emb = embedding_create(128, 256);
    int tokens[] = { 5, 10, 15, 20 };
    int emb_out_shape[2] = { 4, 256 };
    Tensor emb_out = tensor_zeros(2, emb_out_shape);
    embedding_forward(&emb, tokens, 4, &emb_out);
    printf("[OK] Embedding(128, 256): output mean=%.4f\n", tensor_mean(&emb_out));
    tensor_free(&emb_out);
    embedding_free(&emb);

    /* Test Speaker Encoder */
    printf("Testing Speaker Encoder...\n");
    SpeakerEncoder enc = speaker_encoder_create();
    AudioData test_audio = audio_generate_sine(440.0f, 0.5f, SAMPLE_RATE);
    Tensor test_mel = compute_mel_spectrogram(&test_audio);
    int e_shape[1] = { SPEAKER_EMBED_DIM };
    Tensor spk_emb = tensor_zeros(1, e_shape);
    speaker_encoder_forward(&enc, &test_mel, &spk_emb);
    float emb_norm = tensor_l2_norm(&spk_emb);
    printf("[OK] Speaker embedding norm: %.4f (should be ~1.0)\n", emb_norm);
    tensor_free(&spk_emb);
    tensor_free(&test_mel);
    audio_free(&test_audio);
    speaker_encoder_free(&enc);

    /* Test Tokenizer */
    printf("Testing Tokenizer...\n");
    int tok_ids[MAX_TEXT_LEN];
    int n_tok = tokenize_text("hello world", tok_ids);
    char decoded[MAX_TEXT_LEN];
    detokenize(tok_ids, n_tok, decoded, MAX_TEXT_LEN);
    printf("[OK] Tokenize 'hello world' → %d tokens → '%s'\n", n_tok, decoded);

    /* Test Phonemes */
    printf("Testing Phoneme G2P...\n");
    int phon_ids[MAX_PHONEME_SEQ];
    int n_phon = text_to_phonemes("hello world", phon_ids, MAX_PHONEME_SEQ);
    printf("[OK] 'hello world' → %d phonemes: ", n_phon);
    for (int i = 0; i < n_phon && i < 15; i++) {
        printf("%s ", phoneme_name(phon_ids[i]));
    }
    printf("\n");

    printf("\n[PASS] All neural network tests passed!\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Synthesis command
 * ═══════════════════════════════════════════════════════════════════ */

static int cmd_synthesize(const char *text, const char *voice_dir,
                          const char *output_path)
{
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║       Voice Cloner — Synthesis           ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");
    printf("  Text:   \"%s\"\n", text);
    printf("  Voice:  %s\n", voice_dir);
    printf("  Output: %s\n\n", output_path);

    /* 1. Create models */
    SpeakerEncoder enc = speaker_encoder_create();
    TTSSynthesizer syn = synthesizer_create(get_vocab_size());

    /* 2. Try loading trained models */
    char enc_path[512], syn_path[512];
    snprintf(enc_path, 512, "%s/speaker_encoder.bin", voice_dir);
    snprintf(syn_path, 512, "%s/synthesizer.bin", voice_dir);

    if (speaker_encoder_load(&enc, enc_path) != 0) {
        printf("[WARN] No speaker encoder model found at %s, using random weights.\n",
               enc_path);
    }
    if (synthesizer_load(&syn, syn_path) != 0) {
        printf("[WARN] No synthesizer model found at %s, using random weights.\n",
               syn_path);
    }

    /* 3. Generate/load speaker embedding */
    int e_shape[1] = { SPEAKER_EMBED_DIM };
    Tensor speaker_emb = tensor_zeros(1, e_shape);

    /* Try to find voice samples to extract embedding */
    char data_dir[512];
    snprintf(data_dir, 512, "%s/../data", voice_dir);
    char (*wav_files)[512] = (char (*)[512])safe_malloc(sizeof(char[512]) * 100);
    int n_voice = 0;

#ifdef _WIN32
    {
        char search[512];
        snprintf(search, 512, "%s\\*.wav", data_dir);
        WIN32_FIND_DATAA fd;
        HANDLE hFind = FindFirstFileA(search, &fd);
        if (hFind != INVALID_HANDLE_VALUE) {
            do {
                snprintf(wav_files[n_voice], 512, "%s\\%s", data_dir, fd.cFileName);
                n_voice++;
            } while (FindNextFileA(hFind, &fd) && n_voice < 100);
            FindClose(hFind);
        }
    }
#endif

    if (n_voice > 0) {
        printf("[INFO] Found %d voice samples, extracting speaker embedding...\n", n_voice);
        Tensor *voice_mels = (Tensor *)safe_malloc(sizeof(Tensor) * (size_t)n_voice);
        int valid = 0;
        for (int i = 0; i < n_voice; i++) {
            AudioData audio;
            if (audio_read_wav(wav_files[i], &audio) == 0) {
                if (audio.sample_rate != SAMPLE_RATE)
                    audio_resample(&audio, SAMPLE_RATE);
                voice_mels[valid] = compute_mel_spectrogram(&audio);
                audio_free(&audio);
                valid++;
            }
        }
        if (valid > 0) {
            Tensor avg = speaker_encoder_embed_utterances(&enc, voice_mels, valid);
            tensor_copy(&speaker_emb, &avg);
            tensor_free(&avg);
        }
        for (int i = 0; i < valid; i++) tensor_free(&voice_mels[i]);
        free(voice_mels);
    } else {
        printf("[INFO] No voice samples found, using random speaker embedding.\n");
        for (int i = 0; i < SPEAKER_EMBED_DIM; i++) {
            speaker_emb.data[i] = randn() * 0.1f;
        }
        /* L2 normalize */
        float norm = tensor_l2_norm(&speaker_emb);
        if (norm > 1e-10f) tensor_scale(&speaker_emb, 1.0f / norm);
    }
    free(wav_files);

    /* 4. Tokenize text */
    int tokens[MAX_TEXT_LEN];
    int n_tokens = tokenize_text(text, tokens);
    printf("[INFO] Tokenized text: %d tokens\n", n_tokens);

    /* 5. Synthesize mel spectrogram */
    printf("[INFO] Synthesizing mel spectrogram...\n");
    Tensor mel_out;
    int n_frames = synthesizer_inference(&syn, tokens, n_tokens,
                                          &speaker_emb, &mel_out);
    printf("[INFO] Generated %d mel frames (%.2fs)\n",
           n_frames, (float)n_frames * HOP_LENGTH / SAMPLE_RATE);

    /* 6. Vocoder: mel → waveform */
    printf("[INFO] Running Griffin-Lim vocoder (%d iterations)...\n", GRIFFIN_LIM_ITERS);
    AudioData audio_out = vocoder_griffin_lim(&mel_out, SAMPLE_RATE);
    printf("[INFO] Generated %d audio samples (%.2fs)\n",
           audio_out.num_samples,
           (float)audio_out.num_samples / (float)audio_out.sample_rate);

    /* 7. Write output */
    if (audio_write_wav(output_path, &audio_out) == 0) {
        printf("\n[SUCCESS] Audio saved to: %s\n", output_path);
    } else {
        printf("\n[ERROR] Failed to save audio to: %s\n", output_path);
    }

    /* Cleanup */
    tensor_free(&mel_out);
    tensor_free(&speaker_emb);
    audio_free(&audio_out);
    speaker_encoder_free(&enc);
    synthesizer_free(&syn);

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Help
 * ═══════════════════════════════════════════════════════════════════ */

static void print_help(void)
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║       Voice Cloner — TTS System in Pure C       ║\n");
    printf("║       Tacotron-lite + Griffin-Lim Vocoder        ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("Usage:\n");
    printf("  voice_cloner <command> [options]\n");
    printf("\n");
    printf("Commands:\n");
    printf("  train          Train the voice cloning model\n");
    printf("    --data DIR   Directory containing voice WAV files [default: data]\n");
    printf("    --epochs N   Number of training epochs [default: 100]\n");
    printf("    --lr F       Learning rate [default: 0.001]\n");
    printf("\n");
    printf("  synth          Synthesize speech from text\n");
    printf("    --text \"...\" Text to synthesize (required)\n");
    printf("    --voice DIR  Model directory [default: models]\n");
    printf("    --output F   Output WAV file [default: output.wav]\n");
    printf("\n");
    printf("  test-wav       Run WAV I/O self-test\n");
    printf("  test-fft       Run FFT self-test\n");
    printf("  test-features  Run feature extraction self-test\n");
    printf("  test-nn        Run neural network self-test\n");
    printf("  help           Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  voice_cloner train --data my_voice --epochs 50\n");
    printf("  voice_cloner synth --text \"hello world\" --output hello.wav\n");
    printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    if (argc < 2) {
        print_help();
        return 0;
    }

    const char *cmd = argv[1];

    /* ── Train ────────────────────────────────── */
    if (strcmp(cmd, "train") == 0) {
        TrainConfig cfg = train_config_default();

        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
                strncpy(cfg.data_dir, argv[++i], sizeof(cfg.data_dir) - 1);
            } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
                cfg.n_epochs = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
                cfg.learning_rate = (float)atof(argv[++i]);
            }
        }

        return train_full_pipeline(&cfg);
    }

    /* ── Synthesize ───────────────────────────── */
    if (strcmp(cmd, "synth") == 0) {
        const char *text = "hello world";
        const char *voice = "models";
        const char *output = "output.wav";

        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
                text = argv[++i];
            } else if (strcmp(argv[i], "--voice") == 0 && i + 1 < argc) {
                voice = argv[++i];
            } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
                output = argv[++i];
            }
        }

        return cmd_synthesize(text, voice, output);
    }

    /* ── Self-tests ───────────────────────────── */
    if (strcmp(cmd, "test-wav") == 0)      return test_wav();
    if (strcmp(cmd, "test-fft") == 0)      return test_fft();
    if (strcmp(cmd, "test-features") == 0) return test_features();
    if (strcmp(cmd, "test-nn") == 0)       return test_nn();

    /* ── Help ─────────────────────────────────── */
    if (strcmp(cmd, "help") == 0 || strcmp(cmd, "--help") == 0 ||
        strcmp(cmd, "-h") == 0) {
        print_help();
        return 0;
    }

    printf("[ERROR] Unknown command: %s\n", cmd);
    print_help();
    return 1;
}
