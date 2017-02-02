HEADER = $(wildcard include/*.h)
SRC = $(wildcard src/*.c)

python: $(SRC) $(HEADER) setup.py
	python setup.py build

shared: $(SRC) $(HEADER)
	$(CC) src/ccwt.c src/render_png.c -Iinclude -o libccwt.so -Os -fPIC -shared -lm -lpthread -lfftw3 -lfftw3_threads -lpng
