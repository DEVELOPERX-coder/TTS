/*
 * phonemes.h — English phoneme table
 * Voice Cloner TTS System
 */
#ifndef PHONEMES_H
#define PHONEMES_H

#define NUM_PHONEMES    72   /* ARPAbet set + specials */
#define MAX_WORD_LEN    64
#define MAX_PHONEME_SEQ 256

/* Convert a word to phoneme ID sequence (rule-based G2P).
 * Returns number of phoneme IDs written. */
int word_to_phonemes(const char *word, int *phoneme_ids, int max_len);

/* Convert full text to phoneme ID sequence.
 * Returns number of phoneme IDs written. */
int text_to_phonemes(const char *text, int *phoneme_ids, int max_len);

/* Get the string name of a phoneme given its ID. */
const char *phoneme_name(int id);

/* Get total number of phonemes. */
int get_num_phonemes(void);

#endif /* PHONEMES_H */
