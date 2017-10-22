CC=clang
all: mm reduction
reduction:reduction.c
	$(CC) -o $@ $^ -Wall -lpthread
mm:mm.c
	$(CC) -o $@ $^ -Wall -lpthread

.PHONY:clean
clean:
	rm -f mm reduction
