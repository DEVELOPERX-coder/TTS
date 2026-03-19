/*
 * wav.h — WAV file reader/writer (16-bit PCM)
 * Voice Cloner TTS System
 */
#ifndef WAV_H
#define WAV_H

#include <stdint.h>

typedef struct {
    float   *samples;    /* normalized to [-1, 1] */
    int      num_samples;
    int      sample_rate;
    int      channels;
} AudioData;

/* Read a WAV file into AudioData (converts to mono, normalizes to float). */
int audio_read_wav(const char *path, AudioData *out);

/* Write AudioData to a 16-bit PCM WAV file. */
int audio_write_wav(const char *path, const AudioData *audio);

/* Create AudioData from raw float buffer. */
AudioData audio_from_buffer(const float *buf, int n, int sr);

/* Resample to target sample rate (simple linear interpolation). */
void audio_resample(AudioData *audio, int target_sr);

/* Free AudioData samples. */
void audio_free(AudioData *audio);

/* Generate a sine wave for testing. */
AudioData audio_generate_sine(float freq, float duration, int sr);

#endif /* WAV_H */
