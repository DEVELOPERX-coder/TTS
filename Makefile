# ───────────────────────────────────────────────────────────────
# Makefile — Voice Cloner TTS System
# ───────────────────────────────────────────────────────────────

CC       = gcc
CFLAGS   = -std=c11 -Wall -Wextra -Wno-unused-parameter -I src
LDFLAGS  = -lm

# Platform detection
ifeq ($(OS),Windows_NT)
    EXE      = voice_cloner.exe
    RM       = del /Q
    MKDIR    = if not exist models mkdir models & if not exist data mkdir data
else
    EXE      = voice_cloner
    RM       = rm -f
    MKDIR    = mkdir -p models data
endif

# Source files
SRCS = src/main.c \
       src/utils/memory.c \
       src/utils/math_utils.c \
       src/audio/wav.c \
       src/audio/features.c \
       src/audio/vocoder.c \
       src/text/tokenizer.c \
       src/text/phonemes.c \
       src/nn/tensor.c \
       src/nn/layers.c \
       src/nn/attention.c \
       src/nn/encoder.c \
       src/nn/synthesizer.c \
       src/nn/optimizer.c \
       src/nn/train.c

OBJS = $(SRCS:.c=.o)

# ── Targets ──────────────────────────────────────────────────

.PHONY: all clean debug release dirs help

all: dirs $(EXE)
	@echo.
	@echo [BUILD COMPLETE] $(EXE)
	@echo.

$(EXE): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(EXE) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

debug: CFLAGS += -g -O0 -DDEBUG
debug: dirs $(EXE)

release: CFLAGS += -O3 -DNDEBUG -march=native -ffast-math
release: dirs $(EXE)

dirs:
	$(MKDIR)

clean:
	$(RM) $(OBJS) $(EXE) test_sine.wav

help:
	@echo Voice Cloner TTS - Build Targets:
	@echo   make           Build with default flags
	@echo   make debug     Build with debug symbols
	@echo   make release   Build with optimizations
	@echo   make clean     Remove build artifacts
