/*
 * tokenizer.c — Text tokenization
 * Voice Cloner TTS System
 */
#include "tokenizer.h"
#include <string.h>
#include <ctype.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════
 * Text normalization
 * ═══════════════════════════════════════════════════════════════════ */

void text_normalize(const char *input, char *output, int max_len)
{
    int j = 0;
    int prev_space = 0;
    for (int i = 0; input[i] && j < max_len - 1; i++) {
        char c = (char)tolower((unsigned char)input[i]);
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
        if (c == ' ') {
            if (!prev_space && j > 0) {
                output[j++] = ' ';
                prev_space = 1;
            }
        } else {
            output[j++] = c;
            prev_space = 0;
        }
    }
    /* Trim trailing space */
    if (j > 0 && output[j-1] == ' ') j--;
    output[j] = '\0';
}

/* ═══════════════════════════════════════════════════════════════════
 * Character-level tokenization
 * ═══════════════════════════════════════════════════════════════════ */

/* Token mapping:
 * 0 = PAD, 1 = BOS, 2 = EOS
 * 3.. = printable ASCII starting from space (32)
 * char_to_token(c) = c - 32 + 3 for printable chars */

static int char_to_token(char c)
{
    if (c >= 32 && c < 127) return (int)(c - 32) + 3;
    return TOKEN_PAD; /* unknown → pad */
}

static char token_to_char(int t)
{
    if (t < 3) return '\0';
    int c = t - 3 + 32;
    if (c >= 32 && c < 127) return (char)c;
    return '\0';
}

int tokenize_text(const char *text, int *out)
{
    char norm[MAX_TEXT_LEN];
    text_normalize(text, norm, MAX_TEXT_LEN);

    int n = 0;
    out[n++] = TOKEN_BOS;
    for (int i = 0; norm[i] && n < MAX_TEXT_LEN - 1; i++) {
        out[n++] = char_to_token(norm[i]);
    }
    out[n++] = TOKEN_EOS;
    return n;
}

void detokenize(const int *tokens, int n, char *out, int out_size)
{
    int j = 0;
    for (int i = 0; i < n && j < out_size - 1; i++) {
        if (tokens[i] == TOKEN_PAD || tokens[i] == TOKEN_BOS || tokens[i] == TOKEN_EOS)
            continue;
        char c = token_to_char(tokens[i]);
        if (c) out[j++] = c;
    }
    out[j] = '\0';
}

int get_vocab_size(void)
{
    return VOCAB_SIZE;
}
