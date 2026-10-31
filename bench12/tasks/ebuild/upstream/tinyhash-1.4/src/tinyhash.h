#ifndef TINYHASH_H
#define TINYHASH_H
#include <stddef.h>
#include <stdint.h>

#define TINYHASH_VERSION "1.4"

uint64_t tinyhash_fnv1a64(const void *data, size_t len);
uint32_t tinyhash_fold32(uint64_t h);

#endif
