#!/bin/bash
cython -3 --cplus radiation.pyx
g++-9 -shared -fPIC -O2 -march=native -DBOOL_TYPE="signed char" -I../include -I../include_test -I/usr/local/include -I/usr/local/lib/python3.7/site-packages/numpy/core/include -I/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/include/python3.7m -L/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/lib -lpython3.7m -o radiation.so radiation.cpp

