HEADER = $(wildcard include/*.h)
SRC = $(wildcard src/*.c)

python: $(SRC) $(HEADER) setup.py
	ARCHFLAGS='-arch x86_64' python setup.py build

dynamiclib: $(SRC) $(HEADER)
	$(CC) -O3 -lfftw3 -lfftw3_threads -lpthread -lpng -Iinclude -dynamiclib -o libccwt.so src/ccwt.c src/render_png.c
