ccwt.so: ccwt.c
	ARCHFLAGS='-arch x86_64' python setup.py build --build-lib .
