#!/usr/bin/python
#coding=UTF-8

from distutils.core import setup, Extension
import numpy

setup(
    name = 'ccwt',
    version = '1.0',
    description = 'Complex continuous wavelet transformation',
    author = 'Alexander Meißner',
    author_email = 'AlexanderMeissner@gmx.net',
    url = 'https://github.com/lichtso/ccwt',
    ext_modules = [Extension('ccwt',
        include_dirs = ['include', numpy.get_include()],
        libraries = ['fftw3', 'png'],
        sources = ['src/ccwt.c', 'src/render_png.c', 'src/python_api.c']
    )]
)
