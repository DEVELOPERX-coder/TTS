/*
 * memory.h — Memory management utilities
 * Voice Cloner TTS System
 */
#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h>

/* Arena allocator for batch allocations */
typedef struct {
    char   *buf;
    size_t  cap;
    size_t  used;
} Arena;

Arena  arena_create(size_t capacity);
void  *arena_alloc(Arena *a, size_t bytes);
void   arena_reset(Arena *a);
void   arena_destroy(Arena *a);

/* Safe wrappers */
void *safe_malloc(size_t bytes);
void *safe_calloc(size_t count, size_t size);
void *safe_realloc(void *ptr, size_t bytes);

#endif /* MEMORY_H */
