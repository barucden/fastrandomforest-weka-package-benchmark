#!/bin/sh

echo "Cloning the loop version source..."
git clone -q https://github.com/barucden/fastrandomforest-weka-package.git loop-source
echo "Building the loop version..."
mvn -q -f loop-source install -Dmaven.test.skip=true

echo "Building the recursive version benchmark..."
mvn -q -f recursive-version/ package
echo "Building the loop version benchmark..."
mvn -q -f loop-version/ package

echo "Downloading datasets..."
wget -q https://netcologne.dl.sourceforge.net/project/weka/datasets/datasets-UCI/datasets-UCI.jar && jar xf datasets-UCI.jar

export DATA_PATH=`pwd`/UCI/autos.arff

echo "Benchmarking the recursive version..."
java -jar recursive-version/target/benchmarks.jar UCIDatasetBenchmark
echo "Benchmarking the loop version..."
java -jar loop-version/target/benchmarks.jar UCIDatasetBenchmark

export DATA_PATH=`pwd`/UCI/waveform-5000.arff

echo "Benchmarking the recursive version..."
java -jar recursive-version/target/benchmarks.jar UCIDatasetBenchmark
echo "Benchmarking the loop version..."
java -jar loop-version/target/benchmarks.jar UCIDatasetBenchmark
