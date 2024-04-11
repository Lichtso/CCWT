#coding=UTF-8

from setuptools import Extension, setup
import platform, subprocess, numpy

include_dirs = ["include", numpy.get_include()]
library_dirs = []

if platform.system() == "Darwin":
    p = subprocess.run(["which", "-s", "brew"])
    if p.returncode == 0:
        include_dirs.append("/opt/homebrew/include/")
        library_dirs.append("/opt/homebrew/lib/")

setup(
    ext_modules = [Extension("ccwt",
        language = "c",
        extra_compile_args = ["-std=c99"],
        include_dirs = include_dirs,
        library_dirs = library_dirs,
        libraries = ["pthread", "fftw3", "fftw3_threads", "png"],
        sources = ["src/ccwt.c", "src/render_png.c", "src/python_api.c"]
    )]
)
