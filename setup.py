#!/usr/bin/python
#coding=UTF-8

from distutils.core import setup, Extension
import numpy

name = 'ccwt'
version = '0.0.1'
url = 'https://github.com/lichtso/ccwt'

setup(
    name = name,
    version = version,
    license = 'MIT Licence',
    description = 'Complex continuous wavelet transformation',
    platforms = ['Linux', 'Mac OS-X', 'Unix'],
    author = 'Alexander Meißner',
    author_email = 'AlexanderMeissner@gmx.net',
    url = url,
    download_url = url+'/tarball/v'+version,
    data_files = [('headers', ['include/ccwt.h', 'include/render_mode.h'])],
    ext_modules = [Extension(name,
        include_dirs = ['include', numpy.get_include()],
        libraries = ['fftw3', 'fftw3_threads', 'pthread', 'png'],
        sources = ['src/ccwt.c', 'src/render_png.c', 'src/python_api.c']
    )]
)
