# How We Built a Voice Cloner from Scratch in C

> **The complete, beginner-friendly guide.** Read this to understand every piece of the project, then rebuild it yourself.

---

## Table of Contents

1. [The Big Picture — What Are We Building?](#1-the-big-picture)
2. [Prerequisites — What You Need to Know](#2-prerequisites)
3. [Step 1: Memory Management](#3-step-1-memory-management)
4. [Step 2: Math Utilities (FFT, Windowing, Mel Scale)](#4-step-2-math-utilities)
5. [Step 3: Tensors — The Core Data Structure](#5-step-3-tensors)
6. [Step 4: Reading and Writing Audio (WAV Files)](#6-step-4-wav-files)
7. [Step 5: Extracting Features (Mel Spectrogram)](#7-step-5-mel-spectrogram)
8. [Step 6: Text Processing (Tokenizer + Phonemes)](#8-step-6-text-processing)
9. [Step 7: Neural Network Layers](#9-step-7-neural-network-layers)
10. [Step 8: The Attention Mechanism](#10-step-8-attention)
11. [Step 9: The Adam Optimizer](#11-step-9-adam-optimizer)
12. [Step 10: Speaker Encoder](#12-step-10-speaker-encoder)
13. [Step 11: TTS Synthesizer (Tacotron-lite)](#13-step-11-tts-synthesizer)
14. [Step 12: Griffin-Lim Vocoder](#14-step-12-griffin-lim-vocoder)
15. [Step 13: Training Pipeline](#15-step-13-training-pipeline)
16. [Step 14: Main Application (CLI)](#16-step-14-main-application)
17. [Step 15: Build System & Packaging](#17-step-15-build-system)
18. [How It All Connects — End-to-End Flow](#18-end-to-end-flow)
19. [Common Pitfalls & Debugging Tips](#19-common-pitfalls)
20. [Glossary](#20-glossary)

---

## 1. The Big Picture

### What is voice cloning?

Voice cloning = **making a computer speak in someone's voice**. You give it:
- A few seconds of someone talking (the "reference voice")
- A sentence of text

And it produces an audio file of that text spoken in the reference person's voice.

### The 3-Component Architecture

Our system has exactly **3 neural network components** chained together:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   COMPONENT 1: Speaker Encoder                                  │
│   Input:  A person's voice recording (.wav)                     │
│   Output: A "speaker embedding" — 256 numbers that capture      │
│           what this person's voice sounds like                   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   COMPONENT 2: Tacotron-lite Synthesizer                        │
│   Input:  Text + Speaker Embedding                              │
│   Output: A mel spectrogram (image of sound frequencies         │
│           over time)                                            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   COMPONENT 3: Griffin-Lim Vocoder                              │
│   Input:  Mel spectrogram                                       │
│   Output: Actual audio waveform (.wav file you can play)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why C?

Python + PyTorch is the standard for ML. We use C because:
- **Zero dependencies** — everything is from scratch, nothing hidden
- **You learn everything** — every matrix multiply, every FFT butterfly, every gradient
- **Portable** — compiles anywhere with a C compiler
- **Fast** — direct memory control, no garbage collector, no interpreter overhead

---

## 2. Prerequisites

Before reading this guide, you should understand:

| Concept | What you need | Why |
|---------|--------------|-----|
| **C programming** | Pointers, structs, malloc/free, file I/O | This is all in C |
| **Basic linear algebra** | Matrix multiply, vectors, dot product | Neural nets = matrix operations |
| **What a neural network is** | Layers, weights, forward pass | We build NN layers from scratch |
| **What audio is** | Sample rate, waveforms, frequency | We process raw audio |

Don't worry if you're shaky on any of these — this guide explains everything as we go.

---

## 3. Step 1: Memory Management

**Files:** `src/utils/memory.h`, `src/utils/memory.c`

### Why do we need this?

Neural networks allocate a LOT of memory — weight matrices, input/output tensors, intermediate buffers. If any allocation fails silently, we get segfaults that are impossible to debug. So we wrap every malloc.

### What we built:

**1. Safe wrappers:**
```c
void *safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "[FATAL] malloc failed for %zu bytes\n", size);
        exit(1);
    }
    return ptr;
}
```

This is simple but critical. Instead of:
```c
float *data = malloc(1000 * sizeof(float));  // might return NULL!
data[0] = 1.0f;  // CRASH if NULL
```

We do:
```c
float *data = safe_malloc(1000 * sizeof(float));  // exits cleanly if fails
data[0] = 1.0f;  // guaranteed to work
```

We also have `safe_calloc` (allocates zeroed memory) and `safe_realloc` (resizes).

**2. Arena Allocator:**

An arena is a big pre-allocated chunk of memory that you carve sub-allocations from:

```
Arena (1 MB pre-allocated):
┌──────────────────────────────────────────────────┐
│ [alloc1] [alloc2] [alloc3]      free space...    │
└──────────────────────────────────────────────────┘
                              ↑ offset pointer
```

Why? Because calling malloc/free thousands of times per inference step is slow. With an arena, you just bump a pointer. To free everything, reset the pointer to 0. Done.

```c
typedef struct {
    uint8_t *data;      // the big buffer
    size_t   size;      // total capacity
    size_t   offset;    // current allocation point
} Arena;
```

**Key takeaway:** Always wrap memory allocation in C projects. It saves hours of debugging.

---

## 4. Step 2: Math Utilities

**Files:** `src/utils/math_utils.h`, `src/utils/math_utils.c`

This file is the mathematical backbone of the entire project. Everything else builds on these functions.

### 4.1. Complex Numbers

C doesn't have a convenient complex number type for our needs, so:

```c
typedef struct { float re, im; } Complexf;
```

Just a pair of floats. We need these for FFT.

### 4.2. Fast Fourier Transform (FFT)

**What is FFT?** It converts a time-domain signal (audio samples) into a frequency-domain representation (which frequencies are present and how loud).

**Why do we need it?** To create spectrograms — visual representations of audio that neural networks understand.

**The Algorithm (Cooley-Tukey Radix-2):**

The key insight: a size-N DFT can be split into two size-N/2 DFTs:

```
DFT of [x0, x1, x2, x3, x4, x5, x6, x7]
  = DFT of [x0, x2, x4, x6]  (even indices)
  + twiddle_factors × DFT of [x1, x3, x5, x7]  (odd indices)
```

This recursion takes O(N²) → O(N log N). For N=1024, that's ~10,000 ops instead of ~1,000,000.

**Our implementation:**

```c
void fft(Complexf *x, int n) {
    // 1. Bit-reversal permutation
    //    Rearrange elements so the recursive decomposition works in-place
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { /* swap x[i] and x[j] */ }
    }

    // 2. Butterfly operations (bottom-up)
    for (int len = 2; len <= n; len <<= 1) {  // each FFT stage
        float angle = -2.0f * M_PI / len;      // twiddle factor angle
        Complexf wlen = { cosf(angle), sinf(angle) };

        for (int i = 0; i < n; i += len) {     // each group
            Complexf w = {1, 0};                // start with w^0 = 1
            for (int j = 0; j < len/2; j++) {   // each butterfly
                Complexf u = x[i + j];
                Complexf v = complex_mul(w, x[i + j + len/2]);
                x[i + j]         = complex_add(u, v);
                x[i + j + len/2] = complex_sub(u, v);
                w = complex_mul(w, wlen);        // rotate twiddle
            }
        }
    }
}
```

**Visual of one butterfly:**
```
     a ─────┐     ┌───── a + w·b
             │     │
     b ──w──┘     └───── a - w·b
```

**rfft (Real FFT):** Since audio samples are real numbers (not complex), we optimize by packing two real values into one complex number, doing FFT of half size, then unscrambling. This halves computation.

### 4.3. Window Functions

**Problem:** If you just chop audio into chunks and FFT each chunk, you get spectral leakage (frequencies smear). 

**Solution:** Multiply each chunk by a smooth window that tapers to zero at the edges:

```c
void hann_window(float *w, int n) {
    for (int i = 0; i < n; i++)
        w[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (n - 1)));
}
```

The Hann window looks like a bell curve. It's the most common choice for audio analysis.

### 4.4. Mel Scale

**What is the mel scale?** Humans don't hear frequencies linearly. The difference between 100 Hz and 200 Hz sounds huge, but 5000 Hz vs 5100 Hz sounds tiny. The mel scale models this:

```c
float hz_to_mel(float hz)  { return 2595.0f * log10f(1.0f + hz / 700.0f); }
float mel_to_hz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }
```

**Mel filterbank:** A set of triangular filters spaced evenly on the mel scale:

```
Amplitude
    ▲
    │    /\      /\      /\      /\
    │   /  \    /  \    /  \    /  \
    │  /    \  /    \  /    \  /    \
    │ /      \/      \/      \/      \
    └────────────────────────────────────► Frequency (Hz)
      Low freq              High freq
      (wide filters)        (narrow on mel → wide on Hz)
```

Each triangle overlaps with neighbors. We multiply the FFT magnitude by each triangle and sum — this gives us one mel-band energy value per filter.

### 4.5. Activation Functions

Neural networks need nonlinear activation functions:

```c
float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }    // squashes to (0, 1)
float relu_f(float x)    { return x > 0 ? x : 0; }                // kills negatives
// tanh is from math.h                                              // squashes to (-1, 1)
```

### 4.6. Random Number Generation

Neural network weights need random initialization:

```c
float randn(void) {
    // Box-Muller transform: uniform random → normal distribution
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}
```

This gives us standard normal random numbers (mean=0, std=1).

---

## 5. Step 3: Tensors

**Files:** `src/nn/tensor.h`, `src/nn/tensor.c`

### What is a Tensor?

A tensor is just a **multi-dimensional array** with metadata:

```c
typedef struct {
    float *data;        // flat array of numbers
    int    shape[4];    // dimensions, e.g. [3, 256] or [80, 100]
    int    ndim;        // number of dimensions
    int    size;        // total number of elements
    int    owns_data;   // 1 = we should free this, 0 = someone else owns it
} Tensor;
```

Examples:
- A **vector**: shape = [256], ndim = 1
- A **matrix**: shape = [80, 100], ndim = 2 (80 rows, 100 columns)
- A **mel spectrogram**: shape = [80, 200], ndim = 2 (80 mel bands, 200 time frames)

### Data Layout

We store tensors in **row-major** order. For a [3, 4] matrix:

```
Logical:          Memory:
| a b c d |       [a, b, c, d, e, f, g, h, i, j, k, l]
| e f g h |        ↑ row 0    ↑ row 1      ↑ row 2
| i j k l |
```

To access element [row, col]: `data[row * num_cols + col]`

### Key Operations

**Matrix multiplication** is the most important operation. It's what makes neural networks work:

```
C = A × B
[M×K] × [K×N] = [M×N]
```

**Cache-tiled implementation:** Instead of the naive triple loop, we process 32×32 blocks at a time to keep data in CPU cache:

```c
#define TILE 32
for (int i0 = 0; i0 < M; i0 += TILE)
    for (int j0 = 0; j0 < N; j0 += TILE)
        for (int k0 = 0; k0 < K; k0 += TILE)
            for (int i = i0; i < min(i0+TILE, M); i++)
                for (int k = k0; k < min(k0+TILE, K); k++)
                    for (int j = j0; j < min(j0+TILE, N); j++)
                        C[i][j] += A[i][k] * B[k][j];
```

**Why tiling matters:** The naive order accesses B's columns, which are far apart in memory. Tiling keeps both A and B data in the L1/L2 cache. On large matrices, this can be **3-5× faster**.

Other tensor ops we implement:
- `tensor_add`, `tensor_scale` — element-wise operations
- `tensor_softmax` — converts raw scores to probabilities
- `tensor_layer_norm` — normalizes values (stabilizes training)
- `tensor_l2_norm` — vector length
- `tensor_copy` — deep copy values from one tensor to another

---

## 6. Step 4: WAV Files

**Files:** `src/audio/wav.h`, `src/audio/wav.c`

### WAV File Format

WAV is the simplest audio format — uncompressed PCM samples with a header:

```
Bytes 0-3:   "RIFF"
Bytes 4-7:   File size - 8
Bytes 8-11:  "WAVE"
Bytes 12-15: "fmt "
Bytes 16-19: Subchunk size (16 for PCM)
Bytes 20-21: Audio format (1 = PCM)
Bytes 22-23: Number of channels (1=mono, 2=stereo)
Bytes 24-27: Sample rate (e.g. 16000)
Bytes 28-31: Byte rate
Bytes 32-33: Block align
Bytes 34-35: Bits per sample (16)
Bytes 36-39: "data"
Bytes 40-43: Data size
Bytes 44+:   Raw audio samples
```

### Our AudioData Structure

```c
typedef struct {
    float *samples;      // normalized to [-1.0, 1.0]
    int    num_samples;
    int    sample_rate;
    int    channels;
} AudioData;
```

We always convert to:
- **Mono** (if stereo: average left + right channels)
- **Float** (16-bit PCM → divide by 32768.0)
- **Target sample rate** (linear interpolation resampling)

### Reading a WAV File

```c
int audio_read_wav(const char *path, AudioData *audio) {
    FILE *f = fopen(path, "rb");
    // 1. Read and validate header
    // 2. Check format (must be PCM)
    // 3. Read raw samples
    // 4. Convert to float [-1, 1]
    // 5. Mix to mono if stereo
}
```

### Why 16 kHz Sample Rate?

Most TTS systems use 16 kHz because:
- Human speech is mostly below 8 kHz (Nyquist theorem: 16 kHz captures up to 8 kHz)
- Less data to process = faster training
- Good enough quality for understanding speech

---

## 7. Step 5: Mel Spectrogram Extraction

**Files:** `src/audio/features.h`, `src/audio/features.c`

This is the **bridge between raw audio and neural networks**. Neural networks can't understand raw waveforms easily, but they work great with spectrograms.

### The Pipeline

```
Raw Audio → [Window + FFT] → STFT Magnitude → [Mel Filterbank] → [Log Scale] → [Normalize] → Mel Spectrogram
```

### Step-by-Step:

**Step 1: STFT (Short-Time Fourier Transform)**

We can't just FFT the whole audio at once — that would lose all timing information. Instead, we FFT small overlapping windows:

```
Audio signal:
|────────────────────────────────────────────|

Windows (hop = 256 samples, window = 1024 samples):
|========|
    |========|
        |========|
            |========|
                ...

Each window → FFT → one column of the spectrogram
```

Parameters:
- **FFT size** = 1024 → gives us 513 frequency bins (1024/2 + 1)
- **Hop length** = 256 → 75% overlap between windows
- **Window** = Hann window of length 1024

```c
for each time frame t:
    1. Extract 1024 samples starting at t*256
    2. Multiply by Hann window
    3. Compute FFT
    4. Take magnitude: |X[k]| = sqrt(re² + im²)
    5. Store as column t of the spectrogram
```

**Step 2: Apply Mel Filterbank**

Multiply the 513-bin STFT magnitude by our 80 mel filters:

```
     [80 × 513] mel filterbank
  ×  [513 × T]  STFT magnitude
  =  [80 × T]   mel spectrogram
```

Now we have 80 mel-frequency energy values per time frame.

**Step 3: Convert to dB (Log Scale)**

```c
mel_db = 20 * log10(mel_linear + 1e-10)
```

The `1e-10` prevents log(0). The log compression is crucial because:
- Audio energy varies over many orders of magnitude
- Log scale matches human perception (decibels)
- Keeps values in a range that neural networks can handle

**Step 4: Normalize to [0, 1]**

```c
// Clip to [MIN_DB=-100, REF_DB=20], then scale to [0, 1]
mel_normalized = (mel_db - (-100)) / (20 - (-100))
```

Values below -100 dB are silence (clip to 0). Values above 20 dB are peak (clip to 1).

### What Does a Mel Spectrogram Look Like?

```
Mel Band 80 (high freq) │          ░░          ░░
                        │  ░░      ██    ░░    ██
                        │  ██  ░░  ██    ██    ██  ░░
                        │  ██  ██  ██    ██    ██  ██
Mel Band 1 (low freq)   │  ██  ██  ██    ██    ██  ██
                        └────────────────────────────────
                         Time →
```

Dark bands = energy at those frequencies at that time. Speech shows clear horizontal patterns (vowels) and vertical bursts (consonants).

---

## 8. Step 6: Text Processing

**Files:** `src/text/tokenizer.h`, `src/text/tokenizer.c`, `src/text/phonemes.h`, `src/text/phonemes.c`

### Character Tokenization

Neural networks work with numbers, not characters. We map each character to an integer:

```
' ' → 3,  'a' → 68,  'b' → 69, ..., 'z' → 93
'A' → 36, 'B' → 37, ..., 'Z' → 61
'0' → 19, '1' → 20, ..., '9' → 28

Special tokens:
PAD → 0  (padding for batch alignment)
BOS → 1  (beginning of sequence)
EOS → 2  (end of sequence)
```

Tokenization:
```
"Hello" → [1, 39, 72, 79, 79, 82, 2]
          BOS  H   e   l   l   o  EOS
```

### Grapheme-to-Phoneme (G2P)

Characters are a poor representation for pronunciation:
- "though", "through", "thought" — "ough" is pronounced differently each time!

Phonemes are the actual sounds:

```
"hello" → HH EH L OW
"world" → W ER L D
"phone" → F OW N   (not "P-H-O-N-E")
```

We use the **ARPAbet** phoneme set (72 phonemes) — the standard for American English:

| Type | Examples |
|------|----------|
| **Vowels** | AA (father), AE (cat), AH (but), EH (bed), IY (see), OW (go) |
| **Consonants** | B, D, F, G, K, L, M, N, P, R, S, T, V, W, Z |
| **Special** | TH (thin), DH (this), SH (she), CH (church), NG (sing) |

Our G2P converter uses **rules for common patterns**:

```c
if text starts with "th" → TH phoneme
if text starts with "sh" → SH phoneme
if text starts with "ch" → CH phoneme
if text starts with "ph" → F phoneme
if text starts with "ee" → IY phoneme
if word ends with silent "e" → skip the 'e'
// ... more rules
```

This is simplified compared to a full dictionary lookup (like CMUDict) but handles common English words well enough for demonstration.

---

## 9. Step 7: Neural Network Layers

**Files:** `src/nn/layers.h`, `src/nn/layers.c`

This is where we build the **building blocks** that all our models use.

### 9.1. Linear Layer (Fully Connected)

The simplest neural network operation:

```
output = input × Weight + bias
y = x·W + b
```

Where:
- `x` is shape [input_dim]
- `W` is shape [input_dim, output_dim]
- `b` is shape [output_dim]
- `y` is shape [output_dim]

```c
typedef struct {
    Tensor W;  // weight matrix
    Tensor b;  // bias vector
    int in_dim, out_dim;
} LinearLayer;

void linear_forward(LinearLayer *l, const Tensor *x, Tensor *y) {
    // y = x @ W + b
    tensor_matmul(x, &l->W, y);
    tensor_add(y, &l->b);
}
```

**Weight initialization (He init):** We don't start with random weights from any range. He initialization sets:

```
W[i][j] = randn() * sqrt(2.0 / input_dim)
```

This keeps the signal magnitude stable through many layers. Without proper init, deep networks produce exploding or vanishing values.

### 9.2. GRU (Gated Recurrent Unit)

The GRU is a type of **recurrent neural network** — it processes sequences one element at a time while maintaining a "hidden state" (memory):

```
For each time step t:
    z = sigmoid(Wz·x + Uz·h_prev)        ← "update gate": how much to update
    r = sigmoid(Wr·x + Ur·h_prev)        ← "reset gate": how much to forget
    n = tanh(Wn·x + Un·(r ⊙ h_prev))    ← "new content": proposed new info
    h = (1 - z) ⊙ h_prev + z ⊙ n        ← blend old state with new
```

**Intuition:**
- **Update gate z** decides "should I keep the old hidden state or accept new information?"
- **Reset gate r** decides "should I use or ignore the old hidden state when computing new content?"
- The final hidden state is a mix of old (h_prev) and new (n), controlled by z

**Why GRU and not LSTM?** GRU has fewer parameters (2 gates vs 3) and performs comparably for many tasks. Simpler to implement from scratch.

```
Processing the word "hello":
    h0 = zeros(256)
    h1 = GRU('h', h0)   ← sees 'h', updates memory
    h2 = GRU('e', h1)   ← sees 'e', builds on previous
    h3 = GRU('l', h2)   ← sees 'l'
    h4 = GRU('l', h3)   ← sees second 'l'
    h5 = GRU('o', h4)   ← final hidden state encodes "hello"
```

### 9.3. Conv1D (1D Convolution)

Convolutions slide a small filter across the input, capturing local patterns:

```
Input:     [a, b, c, d, e, f, g]
Filter(3): [w1, w2, w3]

Output[0] = w1·a + w2·b + w3·c
Output[1] = w1·b + w2·c + w3·d
Output[2] = w1·c + w2·d + w3·e
...
```

We use **same padding** — pad the input with zeros so the output has the same length.

**Why use convolutions?** They capture local patterns (like character n-grams in text) very efficiently, and they can be computed in parallel (unlike RNNs which are sequential).

### 9.4. Embedding Layer

Maps discrete tokens (integers) to continuous vectors:

```c
typedef struct {
    Tensor W;  // [vocab_size, embed_dim] lookup table
} EmbeddingLayer;

// Forward: just look up the row
void embedding_forward(EmbeddingLayer *e, const int *tokens, int n, Tensor *out) {
    for (int i = 0; i < n; i++) {
        memcpy(out->data + i * embed_dim,
               e->W.data + tokens[i] * embed_dim,
               sizeof(float) * embed_dim);
    }
}
```

Token 42 → copy row 42 of the embedding matrix → that's the vector for token 42.

### 9.5. Layer Normalization

Normalizes the values within each example to have mean=0, variance=1:

```
x_normalized = (x - mean(x)) / sqrt(var(x) + epsilon)
output = gamma * x_normalized + beta
```

Where gamma and beta are learnable parameters. This stabilizes training by preventing the values from drifting.

### 9.6. Prenet

The decoder prenet is a simple 2-layer network with a twist — **dropout is applied even during inference**:

```
Prenet(x) = Dropout(ReLU(Linear2(Dropout(ReLU(Linear1(x))))))
```

**Why dropout at inference?** It adds a small amount of randomness that actually helps the attention mechanism learn better alignments. This is a trick from the original Tacotron 2 paper.

### 9.7. Postnet

5 convolutional layers that refine the mel spectrogram through a **residual connection**:

```
mel_refined = mel_predicted + Postnet(mel_predicted)
```

Each layer: Conv1D(512 channels, kernel 5) → BatchNorm → Tanh (except last layer which has no activation).

The residual connection means the postnet only needs to predict the *correction*, not the entire spectrogram. This makes training much easier.

---

## 10. Step 8: The Attention Mechanism

**Files:** `src/nn/attention.h`, `src/nn/attention.c`

### Why Attention?

The synthesizer needs to turn text into speech. But text and speech are different lengths:
- "Hi" (2 characters) → ~0.5 seconds of audio → ~30 mel frames
- Each mel frame needs to "look at" the right part of the text

Attention is the mechanism that says: **"For this output frame, focus on this part of the input text."**

### Location-Sensitive Attention

We use **location-sensitive attention** (from Chorowski et al.), which is special because it looks at **where attention was focused previously**. This prevents:
- Skipping words (attention jumps forward too fast)
- Repeating words (attention gets stuck)

```
Step 1: Where was attention before?
    cumulative_attention = sum of all previous attention weights
    location_features = Conv1D(cumulative_attention)

Step 2: Score each encoder position
    For each text position j:
        energy[j] = V · tanh(W_q · decoder_state + W_k · encoder[j] + W_f · location[j])

Step 3: Convert scores to probabilities
    attention_weights = softmax(energy)

Step 4: Create context vector
    context = sum(attention_weights[j] * encoder[j])
```

**Visual example:**

```
Text:     "H  E  L  L  O"
           ↓  ↓  ↓  ↓  ↓

Frame 1:  [.9 .1  0  0  0]  ← attention focused on 'H'
Frame 2:  [.8 .2  0  0  0]  ← still mostly 'H'
Frame 3:  [.3 .6 .1  0  0]  ← shifting to 'E'
Frame 4:  [.0 .2 .7 .1  0]  ← now on 'L'
Frame 5:  [.0  0 .1 .8 .1]  ← on second 'L'
Frame 6:  [.0  0  0 .2 .8]  ← on 'O'
```

The attention moves left-to-right through the text, spending multiple frames on each character. This monotonic progression is what makes speech sound natural.

---

## 11. Step 9: The Adam Optimizer

**Files:** `src/nn/optimizer.h`, `src/nn/optimizer.c`

### What is an Optimizer?

Training = adjusting weights to reduce the loss. The optimizer decides **how much** to change each weight and **in which direction**.

### Why Adam?

Adam (Adaptive Moment Estimation) is the most popular optimizer because it:
- Adapts the learning rate for each parameter individually
- Handles noisy gradients well
- Works out of the box for most problems

### The Math

For each parameter θ at step t:

```
1. Compute gradient:    g = ∂Loss/∂θ

2. Update momentum:     m = 0.9 * m + 0.1 * g           (moving average of gradient)
3. Update velocity:     v = 0.999 * v + 0.001 * g²       (moving average of squared gradient)

4. Bias correction:     m̂ = m / (1 - 0.9^t)
                        v̂ = v / (1 - 0.999^t)

5. Update parameter:   θ = θ - lr * m̂ / (√v̂ + 1e-8)
```

**Intuition:**
- **m (momentum):** Which direction has the gradient been pointing? Keep going that way even if the current gradient is noisy.
- **v (velocity):** How big have the gradients been? Scale down parameters with large gradients, scale up parameters with small gradients.
- **Bias correction:** At the start (t=1,2,3...), m and v are biased toward zero. Correction divides by (1 - β^t) to fix this.

### Learning Rate Schedule

We don't use a fixed learning rate. Instead:

```
Phase 1: Warmup (first N steps)
    lr increases linearly from 0 to base_lr
    Why: prevents early gradient explosions when weights are random

Phase 2: Cosine Decay (remaining steps)
    lr = base_lr/2 * (1 + cos(π * progress))
    Why: gradually reduces lr so training settles into a good minimum
```

```
LR
  ↑
  │     /‾‾‾‾‾‾‾‾‾‾‾‾\
  │    /                \
  │   /                  \
  │  /                    \
  │ /                      \_____
  └───────────────────────────────→ Steps
    ↑ warmup    ↑ cosine decay
```

### Gradient Clipping

If gradients become too large (exploding gradients), we clip them:

```c
float total_norm = sqrt(sum of all gradient²);
if (total_norm > max_norm) {
    scale = max_norm / total_norm;
    multiply all gradients by scale;
}
```

This prevents training from diverging.

---

## 12. Step 10: Speaker Encoder

**Files:** `src/nn/encoder.h`, `src/nn/encoder.c`

### Goal

Take any length of audio → produce a fixed-size vector (256 numbers) that captures **who** is speaking, not *what* they're saying.

### Architecture

```
Mel Spectrogram [80 bands × T frames]
        ↓
    GRU Layer 1 (80 → 256)
        ↓
    GRU Layer 2 (256 → 256)
        ↓
    GRU Layer 3 (256 → 256)
        ↓
    Take LAST hidden state → [256]
        ↓
    Linear projection (256 → 256)
        ↓
    ReLU activation
        ↓
    L2 Normalize (make unit length)
        ↓
    Speaker Embedding [256] ← this is the output!
```

### How It Works

The GRU processes the mel spectrogram frame by frame:

```
Frame 1: h1 = GRU(mel_frame_1, h0=zeros)   ← starts with no memory
Frame 2: h2 = GRU(mel_frame_2, h1)          ← builds on previous
Frame 3: h3 = GRU(mel_frame_3, h2)
...
Frame T: hT = GRU(mel_frame_T, h_{T-1})     ← final state = summary of all audio
```

The final hidden state hT has "listened to" the entire audio clip and encodes the speaker's voice characteristics: pitch, timbre, speaking style.

### L2 Normalization

We normalize the embedding to unit length:

```
embedding = embedding / ||embedding||₂
```

This puts all speaker embeddings on the surface of a 256-dimensional sphere. Why?
- Makes **cosine similarity** meaningful for comparing speakers
- Prevents any one speaker from having a "bigger" embedding
- Embeddings become **direction-only** — the direction captures who the speaker is

### Multi-Utterance Embedding

If you have multiple recordings of the same speaker, average the embeddings:

```
e_avg = normalize(mean(e1, e2, e3, ..., eN))
```

This gives a more robust speaker representation by averaging out utterance-specific variations.

---

## 13. Step 11: TTS Synthesizer (Tacotron-lite)

**Files:** `src/nn/synthesizer.h`, `src/nn/synthesizer.c`

This is the **heart** of the system — the neural network that turns text + speaker identity into a mel spectrogram.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│ TEXT ENCODER                                                     │
│                                                                  │
│ Tokens [T] → Embedding [T×256] → 3×Conv1D → BiGRU → [T×256]   │
│                                                + Speaker Embed   │
│                                                                  │
│ Output: "encoder memory" — one vector per input character        │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼ (attention reads from this)
┌──────────────────────────────────────────────────────────────────┐
│ DECODER (autoregressive — generates one frame at a time)         │
│                                                                  │
│ For each output mel frame t:                                     │
│   1. Prenet(previous_mel_frame)     → [128]                     │
│   2. Concat(prenet_out, context)    → [128+256 = 384]           │
│   3. Attention GRU(input, h_attn)   → [512]                     │
│   4. Attention(h_attn, memory)      → context [256], weights    │
│   5. Concat(h_attn, context)        → [512+256 = 768]           │
│   6. Decoder GRU(input, h_dec)      → [512]                     │
│   7. Concat(h_dec, context)         → [768]                     │
│   8. Linear(768 → 80)              → mel_frame [80]             │
│   9. Linear(768 → 1) + sigmoid     → stop_probability          │
│                                                                  │
│ Output mel_frame becomes input to next step (autoregressive)     │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│ POSTNET                                                          │
│                                                                  │
│ 5×Conv1D layers → residual add to mel                           │
│                                                                  │
│ Final mel = predicted mel + postnet correction                   │
└──────────────────────────────────────────────────────────────────┘
```

### Text Encoder Step-by-Step

```
"hello" → [1, 39, 72, 79, 79, 82, 2]     (tokenize)
            ↓
        Embedding: each token → 256-dim vector
            ↓
        [7 × 256] matrix
            ↓
        Conv1D (kernel=5, 256 channels) × 3
        (captures local patterns like "ll", "ell", "llo")
            ↓
        [7 × 256] matrix (same size, richer features)
            ↓
        BiGRU (forward + backward)
        (captures long-range dependencies)
            ↓
        [7 × 256] — the "encoder memory"
            ↓
        + Speaker embedding (broadcast-added)
        (conditions the encoding on who should speak)
            ↓
        [7 × 256] — final encoder output
```

### Decoder Step-by-Step (One Frame)

Let's trace frame t=5 of generating "hello":

```
1. Previous mel frame → Prenet
   [80] → Linear(80→128) → ReLU → Dropout → Linear(128→128) → ReLU → Dropout
   Output: [128]

2. Concatenate with previous context
   [128] ++ [256] = [384]

3. Attention GRU
   GRU(input=[384], h_prev=[512]) → h_attn_new=[512]
   "Where I think I should look in the text, given what I've generated so far"

4. Location-Sensitive Attention
   Query = h_attn_new [512]
   Memory = encoder output [7 × 256]
   Cumulative weights = sum of attention weights from steps 0..4
   → attention_weights = [0, 0, 0.1, 0.7, 0.2, 0, 0]
     (mostly looking at the 4th character 'l')
   → context = weighted sum of encoder memory = [256]

5. Concatenate attention GRU output + context
   [512] ++ [256] = [768]

6. Decoder GRU
   GRU(input=[768], h_prev=[512]) → h_dec_new=[512]
   "Given where I'm looking and my previous state, what should the audio sound like?"

7. Mel projection
   Concat(h_dec, context) = [768]
   Linear(768 → 80) → mel_frame = [80]
   "These 80 numbers are the mel spectrogram for this time step"

8. Stop prediction
   Linear(768 → 1) → sigmoid → stop_prob
   "Should I stop generating? If > 0.5, yes."
```

### Teacher Forcing (Training)

During training, instead of feeding the model its own predictions:
- We feed it the **ground truth** previous mel frame
- This prevents error accumulation (one bad prediction ruining all future ones)
- The model learns faster because it always has correct context

During inference, we use the model's own predictions (since we don't have ground truth).

---

## 14. Step 12: Griffin-Lim Vocoder

**Files:** `src/audio/vocoder.h`, `src/audio/vocoder.c`

### The Problem

We have a mel spectrogram (80-band magnitude information) but we need a waveform (actual audio samples). The challenge: **we threw away phase information** when we computed the spectrogram magnitude. Without phase, we can't do a simple inverse FFT.

### The Griffin-Lim Algorithm

Griffin-Lim **iteratively estimates the phase**:

```
1. Start with random phase
2. Repeat 60 times:
    a. Combine our known magnitude with estimated phase
    b. Inverse FFT → get a time-domain signal
    c. Forward FFT on that signal → get new magnitude + phase
    d. Keep the new phase, throw away the new magnitude
    e. Put our known magnitude back
3. Final inverse FFT → output audio
```

**Why does this work?** Each iteration, the phase estimate gets slightly more consistent. After ~60 iterations, the phase is good enough for intelligible speech.

### Implementation Details

**Step 1: Invert the mel filterbank**

We have [80 mel bands × T frames] but need [513 frequency bins × T frames]:

```c
// Approximate inverse: use transpose of mel filterbank
magnitude = mel_filterbank_transpose × mel_spectrogram
```

This is an approximation (the true inverse is more complex), but works well in practice.

**Step 2: Iterative phase estimation**

```c
for (int iter = 0; iter < 60; iter++) {
    // Forward STFT: signal → complex spectrum
    for each frame:
        window and FFT the signal

    // Replace magnitude, keep phase
    for each time-frequency bin:
        float mag = target_magnitude[bin];
        float phase = atan2(spectrum.im, spectrum.re);
        spectrum.re = mag * cos(phase);
        spectrum.im = mag * sin(phase);

    // Inverse STFT: complex spectrum → signal
    for each frame:
        inverse FFT and overlap-add
}
```

**Step 3: Overlap-Add**

When we inverse FFT each window, we overlap-add them:

```
Window 1: [aaaaaaaaaa]
Window 2:     [bbbbbbbbbb]
Window 3:         [cccccccccc]
Sum:      [aaaaa(a+b)(a+b)(b+c)(b+c)ccccc]
```

Then normalize by the window sum to get the final signal.

### Audio Quality

Griffin-Lim produces **intelligible but not perfect** audio. You'll hear:
- Slight metallic/buzzy quality
- Some phase artifacts (subtle "wobbly" sound)

Neural vocoders (WaveNet, HiFi-GAN) sound much better but are far more complex to implement. Griffin-Lim is the right choice for a from-scratch C implementation because it's fully deterministic and simple.

---

## 15. Step 13: Training Pipeline

**Files:** `src/nn/train.h`, `src/nn/train.c`

### Overview

The training pipeline discovers audio files, prepares them, and trains both models:

```
Phase 1: Train Speaker Encoder
    For each epoch:
        For each WAV file:
            1. Load audio → mel spectrogram
            2. Forward through speaker encoder → embedding
            3. Compare embedding with centroid of all embeddings
            4. Compute contrastive loss
            5. Log loss

Phase 2: Train Synthesizer
    For each epoch:
        For each (text, audio) pair:
            1. Tokenize text
            2. Get speaker embedding from encoder
            3. Run synthesizer with teacher forcing
            4. Compute MSE loss between predicted and target mel
            5. Log loss
```

### Auto-Discovery

The training loop automatically finds `.wav` files in the data directory. On Windows, it uses `FindFirstFile/FindNextFile`; on Unix, `opendir/readdir`.

If no WAV files are found, it generates **synthetic training data** (sine waves at different frequencies) so the pipeline can still demonstrate end-to-end functionality.

### Loss Functions

**Speaker Encoder Loss (Contrastive):**
```
loss = average distance between each utterance embedding and the centroid
     = (1/N) * Σ ||e_i - mean(e)||²
```

Goal: push all embeddings from the same speaker close together.

**Synthesizer Loss (MSE):**
```
loss = (1/T) * Σ_t ||predicted_mel_frame_t - target_mel_frame_t||²
```

Goal: minimize the difference between predicted and real mel spectrograms.

---

## 16. Step 14: Main Application (CLI)

**File:** `src/main.c`

### Commands

```bash
voice_cloner train     --data DIR --epochs N --lr F
voice_cloner synth     --text "..." --voice DIR --output FILE
voice_cloner test-wav
voice_cloner test-fft
voice_cloner test-features
voice_cloner test-nn
voice_cloner help
```

### Synthesis Pipeline (End-to-End)

```
1. Load models       → speaker_encoder.bin, synthesizer.bin
2. Load voice WAVs   → compute mel spectrograms → speaker embedding
3. Tokenize text     → "hello world" → [1, 39, 72, 79, 79, 82, 3, 90, 82, 85, 79, 71, 2]
4. Synthesizer       → tokens + speaker_emb → mel spectrogram [80 × N_frames]
5. Postnet           → refine mel spectrogram
6. Griffin-Lim       → mel spectrogram → waveform
7. Write WAV         → output.wav
```

### Self-Tests

The four built-in tests validate each component:

1. **test-wav**: Generate sine → write WAV → read back → compare samples
2. **test-fft**: FFT of 440 Hz sine → check peak bin is at 440 Hz
3. **test-features**: Mel spectrogram of sine → check shape [80, N] and range [0, 1]
4. **test-nn**: Forward pass through Linear, GRU, Embedding, Speaker Encoder → check finite outputs

---

## 17. Step 15: Build System

**File:** `Makefile`

### Compile Command

The simplest way to build:

```bash
gcc -std=c11 -O2 -Wall -Isrc src/main.c src/utils/*.c src/audio/*.c src/text/*.c src/nn/*.c -o voice_cloner -lm
```

Flags:
- `-std=c11`: Use C11 standard
- `-O2`: Optimization level 2 (good balance of speed and compile time)
- `-Wall`: Enable all warnings
- `-Isrc`: Add `src/` to include search path
- `-lm`: Link math library (for sin, cos, log, etc.)

For maximum speed:
```bash
gcc -std=c11 -O3 -march=native -ffast-math ... -o voice_cloner -lm
```

- `-O3`: Aggressive optimization
- `-march=native`: Use all CPU features available (SSE, AVX, etc.)
- `-ffast-math`: Allow math approximations for speed

---

## 18. How It All Connects — End-to-End Flow

### Training Flow

```
┌───── data/my_voice_01.wav ─────┐
│      data/my_voice_02.wav      │
│      data/my_voice_03.wav      │
└────────────┬───────────────────┘
             ↓
    ┌── Audio Loading ──┐
    │  Read WAV         │
    │  Resample to 16k  │
    │  Convert to float │
    └────────┬──────────┘
             ↓
    ┌── Feature Extract ─┐
    │  STFT → Mel → Log  │
    │  Normalize [0,1]   │
    └────────┬───────────┘
             ↓
    ┌── Speaker Encoder Training ──┐
    │  GRU forward pass            │
    │  Compute embeddings          │
    │  Contrastive loss            │
    │  Save encoder weights        │
    └────────┬─────────────────────┘
             ↓
    ┌── Synthesizer Training ──────┐
    │  Get speaker embedding       │
    │  Tokenize text               │
    │  Teacher-forced decoding     │
    │  MSE loss on mel frames      │
    │  Save synthesizer weights    │
    └──────────────────────────────┘
```

### Inference Flow

```
Input: text="Hello world", voice=models/

    ┌── Load Models ───────────┐
    │  speaker_encoder.bin     │
    │  synthesizer.bin         │
    └──────────┬───────────────┘
               ↓
    ┌── Speaker Embedding ─────┐
    │  Load voice WAVs         │
    │  → Mel spectrograms      │
    │  → GRU → embed → avg    │
    │  → [256] unit vector     │
    └──────────┬───────────────┘
               ↓
    ┌── Text Encoding ─────────┐
    │  "Hello world"           │
    │  → Tokenize: [1,39,..,2]│
    │  → Embed → Conv → GRU   │
    │  → + speaker embed       │
    │  → [T × 256] memory     │
    └──────────┬───────────────┘
               ↓
    ┌── Autoregressive Decoding ┐
    │  Frame 0: start w/ zeros  │
    │  Frame 1: prenet → attn   │
    │           → decode → mel  │
    │  Frame 2: prenet → attn   │
    │           → decode → mel  │
    │  ...                      │
    │  Frame N: stop_prob > 0.5 │
    │           → STOP          │
    └──────────┬────────────────┘
               ↓
    ┌── Postnet ───────────────┐
    │  5×Conv1D refinement     │
    │  mel += postnet(mel)     │
    └──────────┬───────────────┘
               ↓
    ┌── Griffin-Lim Vocoder ───┐
    │  Invert mel filterbank   │
    │  60 iterations of:       │
    │    ISTFT → STFT          │
    │    replace magnitude     │
    │  Final ISTFT → waveform  │
    └──────────┬───────────────┘
               ↓
    ┌── WAV Output ────────────┐
    │  Float → 16-bit PCM      │
    │  Write RIFF header       │
    │  Save to output.wav      │
    └──────────────────────────┘
```

---

## 19. Common Pitfalls & Debugging Tips

### Memory Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| Segfault on startup | Tensor created with wrong shape | Print shape before matmul |
| Segfault during GRU | Wrong input/hidden dimensions | Check `gru_create(input_dim, hidden_dim)` |
| Memory leak | Forgot to `tensor_free()` | Free every tensor you allocate |

### Numerical Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| NaN values after training | Learning rate too high | Reduce LR, add gradient clipping |
| All outputs identical | Weights all zero or too small | Check He initialization |
| Loss not decreasing | Architecture bug | Test each layer individually |
| Mel values outside [0,1] | Bad normalization | Check MIN_DB and REF_DB clipping |

### Audio Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| Pure silence output | Mel spectrogram all zeros | Check synthesizer generates non-zero values |
| Loud clicking | Phase discontinuities in vocoder | Increase Griffin-Lim iterations |
| Wrong pitch | Wrong sample rate | Ensure 16kHz throughout |
| Robotic sound | Normal for Griffin-Lim | Expected — neural vocoder would fix this |

### Debug Strategy

```c
// Add these prints at key points:
printf("Tensor %s: shape=[%d, %d], mean=%.4f, min=%.4f, max=%.4f\n",
       name, t.shape[0], t.shape[1],
       tensor_mean(&t), tensor_min(&t), tensor_max(&t));
```

If you see NaN or Inf anywhere, trace backward to find where it appeared.

---

## 20. Glossary

| Term | Definition |
|------|-----------|
| **Activation function** | Nonlinear function applied after linear operations (ReLU, sigmoid, tanh) |
| **Attention** | Mechanism that lets the decoder focus on different parts of the encoder output |
| **Autoregressive** | Generating output one step at a time, where each step uses the previous output |
| **Bias** | Constant term added after matrix multiplication in a linear layer |
| **Context vector** | Weighted sum of encoder outputs, computed by attention |
| **Conv1D** | 1D convolution — sliding a filter across a sequence |
| **Decoder** | Part of seq2seq that generates the output sequence |
| **Embedding** | Lookup table mapping discrete tokens to continuous vectors |
| **Encoder** | Part of seq2seq that processes the input sequence |
| **Epoch** | One complete pass through all training data |
| **FFT** | Fast Fourier Transform — time domain ↔ frequency domain |
| **GRU** | Gated Recurrent Unit — RNN variant with update and reset gates |
| **Griffin-Lim** | Algorithm to estimate phase from magnitude spectrogram |
| **He initialization** | Weight init scaled by sqrt(2/fan_in) for ReLU networks |
| **Hidden state** | Internal memory of an RNN, updated at each time step |
| **Hop length** | Number of samples between consecutive STFT frames |
| **L2 norm** | Euclidean length of a vector: sqrt(sum of squares) |
| **Layer norm** | Normalizing activations to zero mean and unit variance |
| **Linear layer** | y = x·W + b (matrix multiply + bias) |
| **Loss** | Number measuring how bad the model's predictions are |
| **Mel scale** | Perceptual frequency scale that matches human hearing |
| **Mel spectrogram** | STFT magnitude mapped through mel filterbank, in log scale |
| **Optimizer** | Algorithm that updates weights to minimize loss (e.g., Adam) |
| **Phase** | The timing/position of a wave cycle (0° to 360°) |
| **Phoneme** | A distinct unit of sound in speech (e.g., the "k" in "cat") |
| **Postnet** | Convolutional network that refines mel spectrogram predictions |
| **Prenet** | Small network at decoder input, uses dropout for regularization |
| **Residual connection** | output = input + f(input) — lets the network learn corrections |
| **Sample rate** | Number of audio samples per second (16000 Hz = 16 kHz) |
| **Seq2seq** | Sequence-to-sequence model with encoder and decoder |
| **Softmax** | Converts raw scores to probabilities that sum to 1 |
| **Speaker embedding** | Fixed-size vector representing a speaker's voice identity |
| **Spectrogram** | 2D representation of audio: x=time, y=frequency, color=amplitude |
| **STFT** | Short-Time Fourier Transform — FFT of overlapping windows |
| **Tacotron** | seq2seq TTS architecture by Google (our model is based on this) |
| **Teacher forcing** | Using ground truth as input during training instead of model predictions |
| **Tensor** | Multi-dimensional array (generalization of vectors and matrices) |
| **Tiling** | Processing matrix operations in small blocks for cache efficiency |
| **Token** | Integer representation of a character or phoneme |
| **Twiddle factor** | Complex exponential terms in the FFT butterfly computation |
| **Vocoder** | Converts acoustic features (mel spectrogram) to audio waveform |
| **WAV** | Uncompressed audio file format with RIFF header |
| **Weight** | Learnable parameter in a neural network |
| **Window function** | Smooth taper (e.g., Hann) applied before FFT to reduce spectral leakage |

---

## How to Rebuild This Project Yourself

### Order of Implementation

Build the project in this exact order (each step depends on previous ones):

```
1. memory.h/c        ← used by everything
2. math_utils.h/c    ← FFT, mel, activations
3. tensor.h/c        ← data structure for all NN ops
4. wav.h/c           ← read/write audio
5. features.h/c      ← audio → mel spectrogram
6. tokenizer.h/c     ← text → token integers
7. phonemes.h/c      ← text → phoneme integers
8. layers.h/c        ← Linear, GRU, Conv1D, Embedding
9. attention.h/c     ← location-sensitive attention
10. optimizer.h/c    ← Adam optimizer
11. encoder.h/c      ← speaker encoder network
12. synthesizer.h/c  ← Tacotron-lite
13. vocoder.h/c      ← Griffin-Lim
14. train.h/c        ← training pipeline
15. main.c           ← CLI app
16. Makefile          ← build system
```

### Testing Strategy

After each step, test immediately:
- After step 3: Create a tensor, print values
- After step 4: Read a WAV file, print sample count
- After step 5: Compute mel of a sine wave, print shape
- After step 8: Create a Linear layer, do a forward pass
- After step 11: Create speaker encoder, get embedding, check norm ≈ 1.0
- After step 13: Generate mel from sine, vocoder it, write WAV, listen

**Don't write everything then test.** Test each piece as you build it.

---

*This guide covers every component of the Voice Cloner project. If you read this carefully and implement each step in order, you'll have a fully working voice cloning system built from scratch. Good luck!*
