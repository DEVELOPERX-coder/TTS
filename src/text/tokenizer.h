/*
 * tokenizer.h — Text tokenization
 * Voice Cloner TTS System
 */
#ifndef TOKENIZER_H
#define TOKENIZER_H

#define MAX_TEXT_LEN 1024
#define VOCAB_SIZE   128   /* ASCII printable characters + specials */

/* Special tokens */
#define TOKEN_PAD  0
#define TOKEN_BOS  1
#define TOKEN_EOS  2

/* Tokenize text string to integer sequence.
 * Returns number of tokens written to `out`.
 * out must have space for at least MAX_TEXT_LEN ints. */
int tokenize_text(const char *text, int *out);

/* Convert token IDs back to string. */
void detokenize(const int *tokens, int n, char *out, int out_size);

/* Get vocabulary size. */
int get_vocab_size(void);

/* Normalize text: lowercase, collapse whitespace. */
void text_normalize(const char *input, char *output, int max_len);

#endif /* TOKENIZER_H */
