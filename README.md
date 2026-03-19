# Voice Cloner — TTS System in Pure C

A fully functional text-to-speech voice cloning system written entirely in **C11** with zero external dependencies (only standard C library + math).

## Architecture

```
                        ┌─────────────────┐
  Voice samples (.wav) ──► Speaker Encoder ──► Speaker Embedding (d=256)
                        └─────────────────┘           │
                                                      ▼
                        ┌──────────────────────────────┐
  Text input ──────────► Tacotron-lite Synthesizer     │──► Mel Spectrogram
                        │ (Encoder → Attention → Dec.) │
                        └──────────────────────────────┘
                                                      │
                        ┌─────────────────┐           ▼
  Output audio (.wav) ◄── Griffin-Lim     ◄───────────┘
                        └─────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **Speaker Encoder** | 3-layer GRU → L2-normalized 256-dim embedding |
| **Text Encoder** | Embedding → 3×Conv1D → BiGRU |
| **Decoder** | Prenet → Attention GRU → Decoder GRU → Mel projection |
| **Attention** | Location-sensitive (Chorowski et al.) |
| **Postnet** | 5×Conv1D residual refinement |
| **Vocoder** | Griffin-Lim (60 iterations) |

## Build

### Requirements

- **GCC** (or any C11 compiler)
- **Make** (GNU Make or MSVC NMake)

### Compile

```bash
# Default build
make

# Optimized build
make release

# Debug build
make debug

# Clean
make clean
```

On Windows without Make, compile directly:

```powershell
gcc -std=c11 -O2 -Wall -Isrc src/main.c src/utils/memory.c src/utils/math_utils.c src/audio/wav.c src/audio/features.c src/audio/vocoder.c src/text/tokenizer.c src/text/phonemes.c src/nn/tensor.c src/nn/layers.c src/nn/attention.c src/nn/encoder.c src/nn/synthesizer.c src/nn/optimizer.c src/nn/train.c -o voice_cloner.exe -lm
```

## Usage

### 1. Run Self-Tests

```bash
./voice_cloner test-wav        # WAV I/O test
./voice_cloner test-fft        # FFT test
./voice_cloner test-features   # Mel spectrogram test
./voice_cloner test-nn         # Neural network test
```

### 2. Train on Your Voice

Place your `.wav` voice samples in the `data/` directory, then:

```bash
./voice_cloner train --data data --epochs 100 --lr 0.001
```

The training pipeline will:
1. Train the speaker encoder on your voice samples
2. Train the TTS synthesizer with speaker conditioning

Models are saved to `models/`.

### 3. Synthesize Speech

```bash
./voice_cloner synth --text "Hello, this is my cloned voice!" --output hello.wav
```

With a trained voice model:

```bash
./voice_cloner synth --text "Hello world" --voice models --output output.wav
```

### 4. Help

```bash
./voice_cloner help
```

## Project Structure

```
TTS/
├── src/
│   ├── main.c              — CLI entry point
│   ├── audio/
│   │   ├── wav.c/h         — WAV file I/O (16-bit PCM)
│   │   ├── features.c/h    — STFT, mel spectrogram
│   │   └── vocoder.c/h     — Griffin-Lim vocoder
│   ├── text/
│   │   ├── tokenizer.c/h   — Character tokenizer
│   │   └── phonemes.c/h    — ARPAbet G2P
│   ├── nn/
│   │   ├── tensor.c/h      — N-D tensor operations
│   │   ├── layers.c/h      — Linear, GRU, Conv1D, etc.
│   │   ├── attention.c/h   — Location-sensitive attention
│   │   ├── encoder.c/h     — Speaker encoder
│   │   ├── synthesizer.c/h — Tacotron-lite
│   │   ├── optimizer.c/h   — Adam optimizer
│   │   └── train.c/h       — Training pipeline
│   └── utils/
│       ├── math_utils.c/h  — FFT, mel filterbank, RNG
│       └── memory.c/h      — Arena allocator
├── research/
│   ├── main.tex            — Research paper (IEEE format)
│   └── references.bib      — Bibliography
├── models/                 — Saved model weights
├── data/                   — Voice samples
├── Makefile
└── README.md
```

## Walkthrough

### What Was Built

A **complete TTS voice cloning pipeline** in ~4,100 lines of pure C11:

- **Neural Network Engine** (~1,600 LOC): Tensor ops with cache-tiled matmul, GRU cells, Conv1D, Embedding, LayerNorm, Prenet, Postnet, location-sensitive attention, and Adam optimizer with warmup + cosine decay.
- **Audio Pipeline** (~500 LOC): Cooley-Tukey FFT, Hann windowing, 80-band mel spectrogram extraction, and Griffin-Lim vocoder with 60-iteration phase reconstruction.
- **Text Processing** (~300 LOC): Character-level tokenizer and rule-based English G2P with ARPAbet (72 phonemes, digraph handling, silent-e).
- **Speaker Encoder** (~250 LOC): 3-layer GRU → Linear → ReLU → L2 normalize, producing 256-dim speaker embeddings. Supports multi-utterance averaging.
- **TTS Synthesizer** (~700 LOC): Tacotron-lite with text encoder (Embed→3×Conv→GRU), autoregressive decoder (Prenet→AttnGRU→DecGRU→MelProj), location-sensitive attention, postnet, stop token prediction, and teacher-forced training.
- **Training Pipeline** (~350 LOC): Auto-discovers WAV files, generates synthetic data if none found, trains speaker encoder with contrastive loss, trains synthesizer with teacher forcing, and supports model checkpointing.

### Verification Results

| Test | Status | What It Validates |
|------|--------|-------------------|
| `test-wav` | ✅ PASS | WAV write → read roundtrip, sample accuracy < 0.001 |
| `test-fft` | ✅ PASS | FFT peak bin matches expected frequency (440 Hz) |
| `test-features` | ✅ PASS | Mel spectrogram dims = [80, N], values in [0, 1] |
| `test-nn` | ✅ PASS | Linear, GRU, Embedding, Speaker Encoder, Tokenizer, G2P |

### Key Design Decisions

- **Pure C11** with no external ML libraries — every neural network op is from scratch
- **Cache-tiled matmul** (32×32 blocks) for efficient CPU computation
- **Griffin-Lim vocoder** instead of neural vocoder — keeps the system self-contained
- **Binary model serialization** with magic number verification for checkpointing
- **Cross-platform** — Windows (FindFirstFile) and Unix (opendir) directory enumeration

## Research Paper

A comprehensive LaTeX research paper is included in the `research/` directory. To compile:

```bash
cd research
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The paper covers:
- System architecture with TikZ diagrams
- Speaker encoder (GRU + GE2E loss mathematics)
- Tacotron-lite synthesizer (attention, prenet, postnet)
- Griffin-Lim vocoder algorithm
- Implementation details (memory management, serialization)
- 32 BibTeX references spanning Tacotron, SV2TTS, WaveNet, VITS, and related work

## License

See [LICENSE](LICENSE) for details.