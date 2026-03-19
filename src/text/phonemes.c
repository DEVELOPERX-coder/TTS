/*
 * phonemes.c — English phoneme table + rule-based G2P
 * Voice Cloner TTS System
 */
#include "phonemes.h"
#include <string.h>
#include <ctype.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════
 * ARPAbet phoneme set
 * ═══════════════════════════════════════════════════════════════════ */

static const char *PHONEME_TABLE[] = {
    "<PAD>",  "<BOS>",  "<EOS>",  "<SPC>",   /* 0-3: specials */
    "AA",  "AE",  "AH",  "AO",  "AW",  "AY", /* 4-9: vowels */
    "B",   "CH",  "D",   "DH",  "EH",  "ER", /* 10-15 */
    "EY",  "F",   "G",   "HH",  "IH",  "IY", /* 16-21 */
    "JH",  "K",   "L",   "M",   "N",   "NG", /* 22-27 */
    "OW",  "OY",  "P",   "R",   "S",   "SH", /* 28-33 */
    "T",   "TH",  "UH",  "UW",  "V",   "W",  /* 34-39 */
    "Y",   "Z",   "ZH",                       /* 40-42 */
    "AA0", "AA1", "AA2",                       /* 43-45: stressed vowels */
    "AE0", "AE1", "AE2",                       /* 46-48 */
    "AH0", "AH1", "AH2",                       /* 49-51 */
    "AO0", "AO1", "AO2",                       /* 52-54 */
    "AW0", "AW1", "AW2",                       /* 55-57 */
    "AY0", "AY1", "AY2",                       /* 58-60 */
    "EH0", "EH1", "EH2",                       /* 61-63 */
    "ER0", "ER1", "ER2",                       /* 64-66 */
    "EY0", "EY1", "EY2",                       /* 67-69 */
    "IH0", "IH1",                              /* 70-71 */
};

int get_num_phonemes(void) { return NUM_PHONEMES; }

const char *phoneme_name(int id)
{
    if (id < 0 || id >= NUM_PHONEMES) return "???";
    return PHONEME_TABLE[id];
}

/* ═══════════════════════════════════════════════════════════════════
 * Simple rule-based grapheme-to-phoneme for English
 * (Covers common letter patterns — not exhaustive)
 * ═══════════════════════════════════════════════════════════════════ */

/* Map single letters to default phonemes */
static int letter_phoneme(char c)
{
    switch (c) {
    case 'a': return 6;   /* AH */
    case 'b': return 10;  /* B */
    case 'c': return 23;  /* K */
    case 'd': return 12;  /* D */
    case 'e': return 14;  /* EH */
    case 'f': return 17;  /* F */
    case 'g': return 18;  /* G */
    case 'h': return 19;  /* HH */
    case 'i': return 20;  /* IH */
    case 'j': return 22;  /* JH */
    case 'k': return 23;  /* K */
    case 'l': return 24;  /* L */
    case 'm': return 25;  /* M */
    case 'n': return 26;  /* N */
    case 'o': return 28;  /* OW */
    case 'p': return 30;  /* P */
    case 'q': return 23;  /* K */
    case 'r': return 31;  /* R */
    case 's': return 32;  /* S */
    case 't': return 34;  /* T */
    case 'u': return 36;  /* UH */
    case 'v': return 38;  /* V */
    case 'w': return 39;  /* W */
    case 'x': return 23;  /* K + S approx */
    case 'y': return 40;  /* Y */
    case 'z': return 41;  /* Z */
    default:  return -1;
    }
}

/* Check for common digraphs */
static int try_digraph(const char *s, int *phoneme_id)
{
    struct { const char *g; int p; } digraphs[] = {
        { "th", 35 }, /* TH */
        { "sh", 33 }, /* SH */
        { "ch", 11 }, /* CH */
        { "ng", 27 }, /* NG */
        { "ph", 17 }, /* F */
        { "wh", 39 }, /* W */
        { "ck", 23 }, /* K */
        { "ee", 21 }, /* IY */
        { "oo", 37 }, /* UW */
        { "ou", 8  }, /* AW */
        { "ow", 28 }, /* OW */
        { "ai", 16 }, /* EY */
        { "ay", 16 }, /* EY */
        { "ea", 21 }, /* IY */
        { "oi", 29 }, /* OY */
        { "oy", 29 }, /* OY */
        { NULL, 0  }
    };
    for (int i = 0; digraphs[i].g; i++) {
        if (s[0] == digraphs[i].g[0] && s[1] == digraphs[i].g[1]) {
            *phoneme_id = digraphs[i].p;
            return 1;
        }
    }
    return 0;
}

int word_to_phonemes(const char *word, int *phoneme_ids, int max_len)
{
    int n = 0;
    int len = (int)strlen(word);

    for (int i = 0; i < len && n < max_len; ) {
        /* Try digraph first */
        if (i + 1 < len) {
            int pid;
            if (try_digraph(word + i, &pid)) {
                phoneme_ids[n++] = pid;
                i += 2;
                continue;
            }
        }
        /* Silent 'e' at end */
        if (word[i] == 'e' && i == len - 1 && len > 2) {
            break;
        }
        /* Single letter */
        int p = letter_phoneme(word[i]);
        if (p >= 0) {
            phoneme_ids[n++] = p;
        }
        i++;
    }
    return n;
}

int text_to_phonemes(const char *text, int *phoneme_ids, int max_len)
{
    int n = 0;
    phoneme_ids[n++] = 1; /* BOS */

    char word[MAX_WORD_LEN];
    int wlen = 0;

    for (int i = 0; text[i] && n < max_len - 1; i++) {
        char c = (char)tolower((unsigned char)text[i]);

        if (isalpha((unsigned char)c)) {
            if (wlen < MAX_WORD_LEN - 1) word[wlen++] = c;
        } else {
            if (wlen > 0) {
                word[wlen] = '\0';
                n += word_to_phonemes(word, phoneme_ids + n, max_len - n);
                wlen = 0;
            }
            if (c == ' ' || c == ',' || c == '.' || c == '!' || c == '?') {
                if (n < max_len) phoneme_ids[n++] = 3; /* SPC */
            }
        }
    }

    /* Flush last word */
    if (wlen > 0) {
        word[wlen] = '\0';
        n += word_to_phonemes(word, phoneme_ids + n, max_len - n);
    }

    if (n < max_len) phoneme_ids[n++] = 2; /* EOS */
    return n;
}
