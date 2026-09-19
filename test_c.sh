#!/bin/bash
set -e

echo "Running C KD-Tree Unit Tests..."
gcc -O3 -march=native -o test_bench_kdtree test_bench_kdtree.c -lm
./test_bench_kdtree

echo -e "\nRunning C Haversine Integration Test..."
gcc -O3 -march=native -o bench_haversine_c bench_haversine.c -lm
OUTPUT=$(./bench_haversine_c | grep "checksum=")

if [ -z "$OUTPUT" ]; then
    echo "bench_haversine.c failed to produce expected output"
    exit 1
fi
echo "Integration test passed, bench_haversine.c outputted:"
echo "$OUTPUT"

echo -e "\nRunning C KD-Tree Integration Test..."
gcc -O3 -march=native -o bench_kdtree_c bench_kdtree.c -lm
OUTPUT2=$(./bench_kdtree_c | grep "avg candidates/employee=")

if [ -z "$OUTPUT2" ]; then
    echo "bench_kdtree.c failed to produce expected output"
    exit 1
fi
echo "Integration test passed, bench_kdtree.c outputted:"
echo "$OUTPUT2"

echo -e "\nAll C code tests passed successfully!"
