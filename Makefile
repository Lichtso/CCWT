SRC = $(wildcard src/*.c)

python: $(SRC)
	ARCHFLAGS='-arch x86_64' python setup.py build

dynamiclib: $(SRC)
	$(CC) -O3 -lfftw3 -lpng -Iinclude -dynamiclib -o ccwt.so $<
