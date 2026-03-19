/*
 * train.h — Training loop and utilities
 * Voice Cloner TTS System
 */
#ifndef TRAIN_H
#define TRAIN_H

#include "encoder.h"
#include "synthesizer.h"
#include "optimizer.h"

typedef struct {
    int    n_epochs;
    float  learning_rate;
    int    batch_size;
    int    save_every;     /* save checkpoint every N epochs */
    char   data_dir[512];
    char   model_dir[512];
} TrainConfig;

TrainConfig train_config_default(void);

/* Train speaker encoder on voice samples in data_dir. */
int train_speaker_encoder(SpeakerEncoder *enc, const TrainConfig *cfg);

/* Train TTS synthesizer on paired (text, audio) data. */
int train_synthesizer(TTSSynthesizer *syn, SpeakerEncoder *enc,
                      const TrainConfig *cfg);

/* Full training pipeline: encoder first, then synthesizer. */
int train_full_pipeline(const TrainConfig *cfg);

#endif /* TRAIN_H */
