package org.sample;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class UCIDatasetBenchmark extends FastRandomForestBenchmark {

    private final String DATA_PATH = System.getenv("DATA_PATH");

    private Instances readFile() {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(DATA_PATH));
            ArffLoader.ArffReader arffReader = new ArffLoader.ArffReader(reader);
            Instances data = arffReader.getData();
            data.setClassIndex(data.numAttributes() - 1);
            return data;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    protected Instances getTrainingData() {
        return readFile();
    }

    @Override
    protected Instances getTestingData() {
        return readFile();
    }

}
