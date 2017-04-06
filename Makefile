HEADER = $(wildcard include/*.h)
SRC = $(wildcard src/*.c)

python_build: $(SRC) $(HEADER) setup.py
	python setup.py build

python_install: python_build
	python setup.py install

python_deploy: python_build
	python setup.py sdist upload -r pypi

shared: $(SRC) $(HEADER)
	$(CC) src/ccwt.c src/render_png.c -Iinclude -o libccwt.so -std=c99 -Os -fPIC -shared -lm -lpthread -lfftw3 -lfftw3_threads -lpng
