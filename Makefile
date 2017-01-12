standalone: ccwt.c
	$(CC) -lfftw3 -lpng -o $@ $<
