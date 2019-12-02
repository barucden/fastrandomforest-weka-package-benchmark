package org.sample;

import hr.irb.fastRandomForest.FastRandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.logging.LogManager;

public class ClassifyDataset {

    private static final String DATA_PATH = System.getenv("DATA_PATH");
    private static final float TRAIN_SIZE = 0.5f;

    static {
        //disable annoying logs from some WEKA dependencies
        InputStream stream = ClassifyDataset.class.getResourceAsStream("logging.properties");
        try {
            LogManager.getLogManager().readConfiguration(stream);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Instances readFile() {
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

    public static void main(String[] args) throws Exception {
        Instances instances = readFile();
        FastRandomForest model = new FastRandomForest();

        int train_n = (int) (instances.size() * TRAIN_SIZE);
        Instances train = new Instances(instances, train_n);
        for (int i = 0; i < train_n; i++) {
            train.add(instances.get(i));
        }

        model.buildClassifier(train);

        for (int i = train_n; i < instances.size(); ++i) {
            Instance instance = instances.get(i);
            double newValue = model.classifyInstance(instance);
            System.out.println(newValue);
        }
    }

}
