/*
 * memory.c — Memory management utilities
 * Voice Cloner TTS System
 */
#include "memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Arena allocator ────────────────────────────────────────────── */

Arena arena_create(size_t capacity)
{
    Arena a;
    a.buf  = (char *)malloc(capacity);
    if (!a.buf) {
        fprintf(stderr, "[FATAL] arena_create: failed to allocate %zu bytes\n", capacity);
        exit(EXIT_FAILURE);
    }
    a.cap  = capacity;
    a.used = 0;
    return a;
}

void *arena_alloc(Arena *a, size_t bytes)
{
    /* Align to 16 bytes */
    size_t aligned = (bytes + 15u) & ~(size_t)15u;
    if (a->used + aligned > a->cap) {
        fprintf(stderr, "[FATAL] arena_alloc: out of arena memory (%zu / %zu)\n",
                a->used + aligned, a->cap);
        exit(EXIT_FAILURE);
    }
    void *ptr = a->buf + a->used;
    a->used += aligned;
    return ptr;
}

void arena_reset(Arena *a)
{
    a->used = 0;
}

void arena_destroy(Arena *a)
{
    free(a->buf);
    a->buf  = NULL;
    a->cap  = 0;
    a->used = 0;
}

/* ── Safe wrappers ──────────────────────────────────────────────── */

void *safe_malloc(size_t bytes)
{
    void *p = malloc(bytes);
    if (!p && bytes > 0) {
        fprintf(stderr, "[FATAL] safe_malloc: failed to allocate %zu bytes\n", bytes);
        exit(EXIT_FAILURE);
    }
    return p;
}

void *safe_calloc(size_t count, size_t size)
{
    void *p = calloc(count, size);
    if (!p && count > 0 && size > 0) {
        fprintf(stderr, "[FATAL] safe_calloc: failed to allocate %zu elements\n", count);
        exit(EXIT_FAILURE);
    }
    return p;
}

void *safe_realloc(void *ptr, size_t bytes)
{
    void *p = realloc(ptr, bytes);
    if (!p && bytes > 0) {
        fprintf(stderr, "[FATAL] safe_realloc: failed to reallocate %zu bytes\n", bytes);
        exit(EXIT_FAILURE);
    }
    return p;
}
