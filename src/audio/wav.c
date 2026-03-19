/*
 * wav.c — WAV file reader/writer (16-bit PCM)
 * Voice Cloner TTS System
 */
#include "wav.h"
#include "../utils/memory.h"
#include "../utils/math_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════
 * WAV header structures (packed)
 * ═══════════════════════════════════════════════════════════════════ */

#pragma pack(push, 1)
typedef struct {
    char     riff_id[4];      /* "RIFF" */
    uint32_t file_size;
    char     wave_id[4];      /* "WAVE" */
} RiffHeader;

typedef struct {
    char     chunk_id[4];     /* "fmt " */
    uint32_t chunk_size;
    uint16_t format;          /* 1 = PCM */
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
} FmtChunk;

typedef struct {
    char     chunk_id[4];     /* "data" */
    uint32_t chunk_size;
} DataChunk;
#pragma pack(pop)

/* ═══════════════════════════════════════════════════════════════════
 * Read WAV
 * ═══════════════════════════════════════════════════════════════════ */

int audio_read_wav(const char *path, AudioData *out)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot open WAV file: %s\n", path);
        return -1;
    }

    RiffHeader riff;
    if (fread(&riff, sizeof(RiffHeader), 1, f) != 1 ||
        memcmp(riff.riff_id, "RIFF", 4) != 0 ||
        memcmp(riff.wave_id, "WAVE", 4) != 0) {
        fprintf(stderr, "[ERROR] Invalid WAV header: %s\n", path);
        fclose(f);
        return -1;
    }

    /* Find fmt chunk */
    FmtChunk fmt;
    memset(&fmt, 0, sizeof(fmt));
    int found_fmt = 0, found_data = 0;
    DataChunk data_hdr;
    memset(&data_hdr, 0, sizeof(data_hdr));

    while (!feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size;
        if (fread(chunk_id, 4, 1, f) != 1) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            fseek(f, -8, SEEK_CUR);
            if (fread(&fmt, sizeof(FmtChunk), 1, f) != 1) break;
            /* Skip extra fmt bytes */
            if (fmt.chunk_size > 16) {
                fseek(f, (long)(fmt.chunk_size - 16), SEEK_CUR);
            }
            found_fmt = 1;
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_hdr.chunk_size = chunk_size;
            memcpy(data_hdr.chunk_id, chunk_id, 4);
            found_data = 1;
            break;
        } else {
            /* Skip unknown chunk */
            fseek(f, (long)chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data) {
        fprintf(stderr, "[ERROR] WAV missing fmt/data chunks: %s\n", path);
        fclose(f);
        return -1;
    }

    if (fmt.format != 1) {
        fprintf(stderr, "[ERROR] Only PCM WAV supported (got format=%d): %s\n",
                fmt.format, path);
        fclose(f);
        return -1;
    }

    int bps = fmt.bits_per_sample;
    int channels = fmt.channels;
    int total_samples = (int)(data_hdr.chunk_size / (unsigned)(bps / 8) / (unsigned)channels);

    /* Read raw samples */
    float *samples = (float *)safe_malloc(sizeof(float) * (size_t)total_samples);

    if (bps == 16) {
        int16_t *raw = (int16_t *)safe_malloc(sizeof(int16_t) * (size_t)total_samples * (size_t)channels);
        fread(raw, sizeof(int16_t), (size_t)total_samples * (size_t)channels, f);
        for (int i = 0; i < total_samples; i++) {
            if (channels == 1) {
                samples[i] = (float)raw[i] / 32768.0f;
            } else {
                /* Mix to mono */
                float sum = 0.0f;
                for (int c = 0; c < channels; c++)
                    sum += (float)raw[i * channels + c];
                samples[i] = sum / (32768.0f * (float)channels);
            }
        }
        free(raw);
    } else if (bps == 8) {
        uint8_t *raw = (uint8_t *)safe_malloc((size_t)total_samples * (size_t)channels);
        fread(raw, 1, (size_t)total_samples * (size_t)channels, f);
        for (int i = 0; i < total_samples; i++) {
            if (channels == 1) {
                samples[i] = ((float)raw[i] - 128.0f) / 128.0f;
            } else {
                float sum = 0.0f;
                for (int c = 0; c < channels; c++)
                    sum += (float)raw[i * channels + c] - 128.0f;
                samples[i] = sum / (128.0f * (float)channels);
            }
        }
        free(raw);
    } else {
        fprintf(stderr, "[ERROR] Unsupported bits_per_sample=%d: %s\n", bps, path);
        free(samples);
        fclose(f);
        return -1;
    }

    fclose(f);

    out->samples     = samples;
    out->num_samples = total_samples;
    out->sample_rate = (int)fmt.sample_rate;
    out->channels    = 1; /* always mono output */

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Write WAV
 * ═══════════════════════════════════════════════════════════════════ */

int audio_write_wav(const char *path, const AudioData *audio)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot create WAV file: %s\n", path);
        return -1;
    }

    int n = audio->num_samples;
    uint32_t data_size = (uint32_t)(n * 2); /* 16-bit = 2 bytes per sample */

    RiffHeader riff;
    memcpy(riff.riff_id, "RIFF", 4);
    riff.file_size = 36 + data_size;
    memcpy(riff.wave_id, "WAVE", 4);

    FmtChunk fmt;
    memcpy(fmt.chunk_id, "fmt ", 4);
    fmt.chunk_size     = 16;
    fmt.format         = 1;
    fmt.channels       = 1;
    fmt.sample_rate    = (uint32_t)audio->sample_rate;
    fmt.byte_rate      = (uint32_t)audio->sample_rate * 2;
    fmt.block_align    = 2;
    fmt.bits_per_sample = 16;

    DataChunk data_hdr;
    memcpy(data_hdr.chunk_id, "data", 4);
    data_hdr.chunk_size = data_size;

    fwrite(&riff, sizeof(RiffHeader), 1, f);
    fwrite(&fmt,  sizeof(FmtChunk), 1, f);
    fwrite(&data_hdr, sizeof(DataChunk), 1, f);

    /* Convert float → int16 and write */
    int16_t *pcm = (int16_t *)safe_malloc(sizeof(int16_t) * (size_t)n);
    for (int i = 0; i < n; i++) {
        float s = audio->samples[i];
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        pcm[i] = (int16_t)(s * 32767.0f);
    }
    fwrite(pcm, sizeof(int16_t), (size_t)n, f);
    free(pcm);

    fclose(f);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Utility functions
 * ═══════════════════════════════════════════════════════════════════ */

AudioData audio_from_buffer(const float *buf, int n, int sr)
{
    AudioData a;
    a.samples = (float *)safe_malloc(sizeof(float) * (size_t)n);
    memcpy(a.samples, buf, sizeof(float) * (size_t)n);
    a.num_samples = n;
    a.sample_rate = sr;
    a.channels    = 1;
    return a;
}

void audio_resample(AudioData *audio, int target_sr)
{
    if (audio->sample_rate == target_sr) return;

    double ratio = (double)target_sr / (double)audio->sample_rate;
    int new_len = (int)((double)audio->num_samples * ratio);
    float *new_buf = (float *)safe_calloc((size_t)new_len, sizeof(float));

    for (int i = 0; i < new_len; i++) {
        double src_pos = (double)i / ratio;
        int idx = (int)src_pos;
        double frac = src_pos - (double)idx;
        if (idx + 1 < audio->num_samples) {
            new_buf[i] = (float)((1.0 - frac) * (double)audio->samples[idx] +
                                  frac * (double)audio->samples[idx + 1]);
        } else if (idx < audio->num_samples) {
            new_buf[i] = audio->samples[idx];
        }
    }

    free(audio->samples);
    audio->samples     = new_buf;
    audio->num_samples = new_len;
    audio->sample_rate = target_sr;
}

void audio_free(AudioData *audio)
{
    if (audio->samples) {
        free(audio->samples);
        audio->samples = NULL;
    }
    audio->num_samples = 0;
}

AudioData audio_generate_sine(float freq, float duration, int sr)
{
    int n = (int)((float)sr * duration);
    AudioData a;
    a.samples = (float *)safe_malloc(sizeof(float) * (size_t)n);
    a.num_samples = n;
    a.sample_rate = sr;
    a.channels    = 1;

    for (int i = 0; i < n; i++) {
        a.samples[i] = 0.5f * sinf(2.0f * (float)M_PI * freq * (float)i / (float)sr);
    }
    return a;
}
