# ziggy

Things are going a little caca.

To build extensions:
```bash
python setup.py install --user
```

To build testing cpp get `libtorch` and run:
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/home/doug/mlai/ziggy/cpp/libtorch ..
cmake --build . --config Release
```
