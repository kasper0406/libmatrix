#!/usr/bin/env bash

set -euo pipefail

nvcc -ccbin clang++ -Xcompiler -fPIC -shared -rdc=true -m64 -o libmatrix.o libmatrix.cu
ar rcs libmatrix.a libmatrix.o

clang++ tester.cpp libmatrix.a -o tester
