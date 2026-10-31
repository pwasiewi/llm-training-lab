# tinyhash — upstream Makefile
PREFIX = /usr/local
CC = gcc
CFLAGS = -O3 -march=native -Wall -fPIC
LDFLAGS = -Wl,-rpath,$(CURDIR)
VERSION = 1.4
SONAME = libtinyhash.so.1

all: libtinyhash.so.$(VERSION) libtinyhash.a thash

src/%.o: src/%.c src/tinyhash.h
	$(CC) $(CFLAGS) -c -o $@ $<

tests/%.o: tests/%.c src/tinyhash.h
	$(CC) $(CFLAGS) -c -o $@ $<

libtinyhash.so.$(VERSION): src/tinyhash.o
	$(CC) -shared -Wl,-soname,$(SONAME) -o $@ $^
	ln -sf $@ $(SONAME)
	ln -sf $(SONAME) libtinyhash.so

libtinyhash.a: src/tinyhash.o
	ar rcs $@ $^

thash: src/thash.o libtinyhash.so.$(VERSION)
	$(CC) $(LDFLAGS) -o $@ src/thash.o -L. -ltinyhash

tests/check: tests/check.o libtinyhash.so.$(VERSION)
	$(CC) $(LDFLAGS) -o $@ tests/check.o -L. -ltinyhash

check: tests/check
	LD_LIBRARY_PATH=. ./tests/check

install: all
	install -d $(PREFIX)/bin $(PREFIX)/lib $(PREFIX)/include $(PREFIX)/share/man/man1 $(PREFIX)/share/doc/tinyhash
	install -s -m755 thash $(PREFIX)/bin/thash
	install -m755 libtinyhash.so.$(VERSION) $(PREFIX)/lib/
	ln -sf $(PREFIX)/lib/libtinyhash.so.$(VERSION) $(PREFIX)/lib/$(SONAME)
	ln -sf $(PREFIX)/lib/$(SONAME) $(PREFIX)/lib/libtinyhash.so
	install -m644 libtinyhash.a $(PREFIX)/lib/
	install -m644 src/tinyhash.h $(PREFIX)/include/
	install -m644 doc/thash.1 $(PREFIX)/share/man/man1/
	install -m644 README NEWS $(PREFIX)/share/doc/tinyhash/

clean:
	rm -f src/*.o tests/*.o tests/check thash libtinyhash.*

.PHONY: all check install clean
