SRC = $(wildcard src/*.c)

python: $(SRC)
	ARCHFLAGS='-arch x86_64' python setup.py build

dynamiclib: $(SRC)
	$(CC) -O3 -lfftw3 -lfftw3_threads -lpthread -lpng -Iinclude -dynamiclib -o libccwt.so $<
