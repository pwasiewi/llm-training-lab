#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include "tinyhash.h"

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "usage: thash STRING...\n");
		return 2;
	}
	for (int i = 1; i < argc; i++)
		printf("%016" PRIx64 "  %s\n", tinyhash_fnv1a64(argv[i], strlen(argv[i])), argv[i]);
	return 0;
}
