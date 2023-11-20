## Bulid

```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Usage

```sh
./mlp -d ../data -l 0.01 -e 10 -h 10
./mlp -d ../data -l 0.01 -e 1 -h 10        // testing
```