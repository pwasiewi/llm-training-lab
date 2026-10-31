#include "tinyhash.h"

uint64_t tinyhash_fnv1a64(const void *data, size_t len)
{
	const unsigned char *p = data;
	uint64_t h = 0xcbf29ce484222325ULL;
	for (size_t i = 0; i < len; i++) {
		h ^= p[i];
		h *= 0x100000001b3ULL;
	}
	return h;
}

uint32_t tinyhash_fold32(uint64_t h)
{
	return (uint32_t)(h ^ (h >> 32));
}
