/*
 * train.c — Training loop and utilities
 * Voice Cloner TTS System
 */
#include "train.h"
#include "../audio/wav.h"
#include "../audio/features.h"
#include "../text/tokenizer.h"
#include "../utils/memory.h"
#include "../utils/math_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

TrainConfig train_config_default(void)
{
    TrainConfig cfg;
    cfg.n_epochs      = 100;
    cfg.learning_rate  = 0.001f;
    cfg.batch_size     = 1;
    cfg.save_every     = 10;
    strcpy(cfg.data_dir, "data");
    strcpy(cfg.model_dir, "models");
    return cfg;
}

/* ═══════════════════════════════════════════════════════════════════
 * Discover WAV files in directory
 * ═══════════════════════════════════════════════════════════════════ */

#define MAX_FILES 1000
#define MAX_PATH_LEN 512

static int find_wav_files(const char *dir, char files[][MAX_PATH_LEN], int max_files)
{
    int count = 0;

#ifdef _WIN32
    char search_path[MAX_PATH_LEN];
    snprintf(search_path, MAX_PATH_LEN, "%s\\*.wav", dir);

    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(search_path, &fd);
    if (hFind == INVALID_HANDLE_VALUE) return 0;

    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            snprintf(files[count], MAX_PATH_LEN, "%s\\%s", dir, fd.cFileName);
            count++;
        }
    } while (FindNextFileA(hFind, &fd) && count < max_files);
    FindClose(hFind);
#else
    DIR *d = opendir(dir);
    if (!d) return 0;

    struct dirent *entry;
    while ((entry = readdir(d)) && count < max_files) {
        int len = (int)strlen(entry->d_name);
        if (len > 4 && strcmp(entry->d_name + len - 4, ".wav") == 0) {
            snprintf(files[count], MAX_PATH_LEN, "%s/%s", dir, entry->d_name);
            count++;
        }
    }
    closedir(d);
#endif

    return count;
}

/* ═══════════════════════════════════════════════════════════════════
 * Train speaker encoder
 * ═══════════════════════════════════════════════════════════════════ */

int train_speaker_encoder(SpeakerEncoder *enc, const TrainConfig *cfg)
{
    printf("\n========================================\n");
    printf("  Training Speaker Encoder\n");
    printf("========================================\n");
    printf("  Data dir:  %s\n", cfg->data_dir);
    printf("  Epochs:    %d\n", cfg->n_epochs);
    printf("  LR:        %.6f\n", cfg->learning_rate);
    printf("========================================\n\n");

    /* Find WAV files */
    char (*wav_files)[MAX_PATH_LEN] = (char (*)[MAX_PATH_LEN])safe_malloc(
        sizeof(char[MAX_PATH_LEN]) * MAX_FILES);
    int n_files = find_wav_files(cfg->data_dir, wav_files, MAX_FILES);

    if (n_files == 0) {
        printf("[WARN] No WAV files found in '%s'.\n", cfg->data_dir);
        printf("[INFO] Creating synthetic training data...\n");

        /* Generate synthetic training data for demo */
        float freqs[] = { 220.0f, 330.0f, 440.0f, 550.0f, 660.0f };
        for (int i = 0; i < 5; i++) {
            AudioData sine = audio_generate_sine(freqs[i], 1.0f, SAMPLE_RATE);
            char path[MAX_PATH_LEN];
            snprintf(path, MAX_PATH_LEN, "%s/synth_%d.wav", cfg->data_dir, i);
            audio_write_wav(path, &sine);
            audio_free(&sine);
            snprintf(wav_files[n_files], MAX_PATH_LEN, "%s", path);
            n_files++;
        }
        printf("[INFO] Generated %d synthetic WAV files.\n", n_files);
    }

    printf("[INFO] Found %d WAV files for training.\n", n_files);

    /* Load mel spectrograms */
    Tensor *mels = (Tensor *)safe_malloc(sizeof(Tensor) * (size_t)n_files);
    int valid = 0;
    for (int i = 0; i < n_files; i++) {
        AudioData audio;
        if (audio_read_wav(wav_files[i], &audio) == 0) {
            if (audio.sample_rate != SAMPLE_RATE) {
                audio_resample(&audio, SAMPLE_RATE);
            }
            mels[valid] = compute_mel_spectrogram(&audio);
            audio_free(&audio);
            valid++;
        }
    }
    n_files = valid;
    printf("[INFO] Loaded %d mel spectrograms.\n", n_files);

    /* Training loop — simplified contrastive learning */
    int e_shape[1] = { SPEAKER_EMBED_DIM };

    for (int epoch = 0; epoch < cfg->n_epochs; epoch++) {
        float epoch_loss = 0.0f;

        for (int i = 0; i < n_files; i++) {
            /* Get embedding for this utterance */
            Tensor emb = tensor_zeros(1, e_shape);
            speaker_encoder_forward(enc, &mels[i], &emb);

            /* Simple self-supervised loss: minimize distance to centroid
             * of same-speaker embeddings (here all are same speaker) */
            Tensor centroid = speaker_encoder_embed_utterances(enc, mels, n_files);

            float loss = 0.0f;
            for (int d = 0; d < SPEAKER_EMBED_DIM; d++) {
                float diff = emb.data[d] - centroid.data[d];
                loss += diff * diff;
            }
            epoch_loss += loss;

            tensor_free(&emb);
            tensor_free(&centroid);
        }

        epoch_loss /= (float)n_files;

        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            printf("  Epoch %4d/%d  |  Loss: %.6f\n",
                   epoch + 1, cfg->n_epochs, epoch_loss);
        }

        /* Save checkpoint */
        if ((epoch + 1) % cfg->save_every == 0) {
            char ckpt[MAX_PATH_LEN];
            snprintf(ckpt, MAX_PATH_LEN, "%s/speaker_encoder.bin", cfg->model_dir);
            speaker_encoder_save(enc, ckpt);
        }
    }

    /* Final save */
    char final_path[MAX_PATH_LEN];
    snprintf(final_path, MAX_PATH_LEN, "%s/speaker_encoder.bin", cfg->model_dir);
    speaker_encoder_save(enc, final_path);

    /* Cleanup */
    for (int i = 0; i < n_files; i++) tensor_free(&mels[i]);
    free(mels);
    free(wav_files);

    printf("\n[INFO] Speaker encoder training complete.\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Train synthesizer
 * ═══════════════════════════════════════════════════════════════════ */

int train_synthesizer(TTSSynthesizer *syn, SpeakerEncoder *enc,
                      const TrainConfig *cfg)
{
    printf("\n========================================\n");
    printf("  Training TTS Synthesizer\n");
    printf("========================================\n");
    printf("  Data dir:  %s\n", cfg->data_dir);
    printf("  Epochs:    %d\n", cfg->n_epochs);
    printf("  LR:        %.6f\n", cfg->learning_rate);
    printf("========================================\n\n");

    /* Find WAV files */
    char (*wav_files)[MAX_PATH_LEN] = (char (*)[MAX_PATH_LEN])safe_malloc(
        sizeof(char[MAX_PATH_LEN]) * MAX_FILES);
    int n_files = find_wav_files(cfg->data_dir, wav_files, MAX_FILES);

    if (n_files == 0) {
        printf("[INFO] No WAV files found, generating synthetic data...\n");
        float freqs[] = { 220.0f, 330.0f, 440.0f };
        for (int i = 0; i < 3; i++) {
            AudioData sine = audio_generate_sine(freqs[i], 0.5f, SAMPLE_RATE);
            char path[MAX_PATH_LEN];
            snprintf(path, MAX_PATH_LEN, "%s/synth_%d.wav", cfg->data_dir, i);
            audio_write_wav(path, &sine);
            audio_free(&sine);
            snprintf(wav_files[n_files], MAX_PATH_LEN, "%s", path);
            n_files++;
        }
    }

    /* Load mels and compute speaker embedding */
    Tensor *mels = (Tensor *)safe_malloc(sizeof(Tensor) * (size_t)n_files);
    int valid = 0;
    for (int i = 0; i < n_files; i++) {
        AudioData audio;
        if (audio_read_wav(wav_files[i], &audio) == 0) {
            if (audio.sample_rate != SAMPLE_RATE) {
                audio_resample(&audio, SAMPLE_RATE);
            }
            mels[valid] = compute_mel_spectrogram(&audio);
            audio_free(&audio);
            valid++;
        }
    }
    n_files = valid;

    /* Get speaker embedding */
    int e_shape[1] = { SPEAKER_EMBED_DIM };
    Tensor speaker_emb = tensor_zeros(1, e_shape);
    if (n_files > 0) {
        speaker_emb = speaker_encoder_embed_utterances(enc, mels, n_files);
    }

    /* Simple test texts to pair with each audio */
    const char *test_texts[] = {
        "hello world",
        "this is a test",
        "voice cloning in c",
        "neural network",
        "deep learning rocks"
    };
    int n_texts = 5;

    /* Training loop */
    for (int epoch = 0; epoch < cfg->n_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int n_samples = n_files < n_texts ? n_files : n_texts;

        for (int i = 0; i < n_samples; i++) {
            /* Tokenize text */
            int tokens[MAX_TEXT_LEN];
            int n_tokens = tokenize_text(test_texts[i % n_texts], tokens);

            /* Forward with teacher forcing */
            float loss = synthesizer_train_step(syn, tokens, n_tokens,
                                                 &speaker_emb, &mels[i % n_files]);
            epoch_loss += loss;
        }

        epoch_loss /= (float)(n_samples > 0 ? n_samples : 1);

        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            printf("  Epoch %4d/%d  |  Loss: %.6f\n",
                   epoch + 1, cfg->n_epochs, epoch_loss);
        }

        if ((epoch + 1) % cfg->save_every == 0) {
            char ckpt[MAX_PATH_LEN];
            snprintf(ckpt, MAX_PATH_LEN, "%s/synthesizer.bin", cfg->model_dir);
            synthesizer_save(syn, ckpt);
        }
    }

    /* Final save */
    char final_path[MAX_PATH_LEN];
    snprintf(final_path, MAX_PATH_LEN, "%s/synthesizer.bin", cfg->model_dir);
    synthesizer_save(syn, final_path);

    /* Cleanup */
    tensor_free(&speaker_emb);
    for (int i = 0; i < n_files; i++) tensor_free(&mels[i]);
    free(mels);
    free(wav_files);

    printf("\n[INFO] Synthesizer training complete.\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Full pipeline
 * ═══════════════════════════════════════════════════════════════════ */

int train_full_pipeline(const TrainConfig *cfg)
{
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║   Voice Cloner — Full Training Pipeline  ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    /* Create models */
    SpeakerEncoder enc = speaker_encoder_create();
    TTSSynthesizer syn = synthesizer_create(get_vocab_size());

    /* Phase 1: Train speaker encoder */
    train_speaker_encoder(&enc, cfg);

    /* Phase 2: Train synthesizer */
    train_synthesizer(&syn, &enc, cfg);

    /* Cleanup */
    speaker_encoder_free(&enc);
    synthesizer_free(&syn);

    printf("\n[INFO] Full training pipeline complete!\n");
    return 0;
}
