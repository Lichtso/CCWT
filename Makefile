HEADER = $(wildcard include/*.h)
SRC = $(wildcard src/*.c)

python_build: $(SRC) $(HEADER) setup.py
	python3 -m build

python_install: python_build
	python3 -m pip install .

python_pack:
	python3 -m build --sdist

python_deploy: python_pack
	python3 -m twine upload dist/*

shared: $(SRC) $(HEADER)
	$(CC) src/ccwt.c src/render_png.c -Iinclude -o libccwt.so -std=c99 -Os -fPIC -shared -lm -lpthread -lfftw3 -lfftw3_threads -lpng
