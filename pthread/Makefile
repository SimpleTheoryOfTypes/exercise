CC=clang
all: mm test
test:test.c
	$(CC) -o $@ $^ -Wall -lpthread
mm:mm.c
	$(CC) -o $@ $^ -Wall -lpthread

.PHONY:clean
clean:
	rm -f mm test
