#coding=UTF-8

from distutils.core import setup, Extension
import numpy

name = 'ccwt'
version = '0.0.7'
url = 'https://github.com/lichtso/ccwt'

import os
os.environ['ARCHFLAGS'] = '-arch x86_64'
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
    ext_modules = [Extension(name,
        language = 'c',
        extra_compile_args = ['-std=c99'],
        include_dirs = ['include', numpy.get_include()],
        libraries = ['pthread', 'fftw3', 'fftw3_threads', 'png'],
        sources = ['src/ccwt.c', 'src/render_png.c', 'src/python_api.c']
    )]
)
