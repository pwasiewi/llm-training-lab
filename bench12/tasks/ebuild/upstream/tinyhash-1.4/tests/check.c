#include <stdio.h>
#include <string.h>
#include "../src/tinyhash.h"

static int fails;
static void expect(const char *s, unsigned long long want)
{
	unsigned long long got = tinyhash_fnv1a64(s, strlen(s));
	if (got != want) {
		printf("FAIL %s: %016llx != %016llx\n", s, got, want);
		fails++;
	}
}

int main(void)
{
	expect("", 0xcbf29ce484222325ULL);
	expect("a", 0xaf63dc4c8601ec8cULL);
	expect("foobar", 0x85944171f73967e8ULL);
	expect("hello", 0xa430d84680aabd0bULL);
	if (tinyhash_fold32(0x0000000100000001ULL) != 0) { puts("FAIL fold32"); fails++; }
	if (strcmp(TINYHASH_VERSION, "1.4") != 0) { puts("FAIL version"); fails++; }
	if (fails)
		return 1;
	printf("tinyhash: all 6 checks passed\n");
	return 0;
}
