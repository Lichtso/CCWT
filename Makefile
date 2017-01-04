cwt: cwt.cpp
	$(CXX) -lfftw3 -lpng -o $@ $<
