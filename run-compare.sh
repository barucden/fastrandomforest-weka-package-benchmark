#!/bin/sh

echo "Cloning the loop version source..."
git clone -q https://github.com/barucden/fastrandomforest-weka-package.git loop-source
echo "Building the loop version..."
mvn -q -f loop-source install -Dmaven.test.skip=true

echo "Building the recursive version benchmark..."
mvn -q -f recursive-version-classify/ package
echo "Building the loop version benchmark..."
mvn -q -f loop-version-classify/ package

echo "Downloading datasets..."
wget -q https://netcologne.dl.sourceforge.net/project/weka/datasets/datasets-UCI/datasets-UCI.jar && jar xf datasets-UCI.jar

for data in `ls UCI/`
do
    export DATA_PATH=`pwd`/UCI/$data

    java -jar recursive-version-classify/target/classify.jar > recursive.out
    java -jar loop-version-classify/target/classify.jar > loop.out

    cmp --silent recursive.out loop.out && echo "[$data] Predictions are the same" || echo "[$data] Predictions differ!!!!!!! <------"
done
