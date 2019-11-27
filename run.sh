#!/bin/sh

echo "Cloning the loop version source..."
git clone -q https://github.com/barucden/fastrandomforest-weka-package.git loop-source
echo "Building the loop version..."
mvn -q -f loop-source install -Dmaven.test.skip=true

echo "Building the recursive version benchmark..."
mvn -q -f recursive-version/ package
echo "Building the loop version benchmark..."
mvn -q -f loop-version/ package

echo "Benchmarking the recursive version..."
java -jar recursive-version/target/benchmarks.jar NumericDatasetBenchmark
java -jar recursive-version/target/benchmarks.jar NominalDatasetBenchmark
echo "Benchmarking the loop version..."
java -jar loop-version/target/benchmarks.jar NumericDatasetBenchmark
java -jar loop-version/target/benchmarks.jar NominalDatasetBenchmark
