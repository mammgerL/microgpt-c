CC=clang
CFLAGS=-O3 -Wall -Wextra -std=c11 -DACCELERATE_NEW_LAPACK
LDFLAGS=-framework Accelerate -lm

all: microgpt_mac

microgpt_mac: microgpt_mac.c
	$(CC) $(CFLAGS) microgpt_mac.c -o microgpt_mac $(LDFLAGS)

run: microgpt_mac
	./microgpt_mac

clean:
	rm -f microgpt_mac
