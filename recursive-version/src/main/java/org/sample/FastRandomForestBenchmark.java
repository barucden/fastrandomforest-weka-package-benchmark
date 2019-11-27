package org.sample;

import hr.irb.fastRandomForest.FastRandomForest;
import org.openjdk.jmh.annotations.*;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.LogManager;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 1)
@Measurement(iterations = 10)
@Fork(value = 1)
public abstract class FastRandomForestBenchmark {

    static {
        //disable annoying logs from some WEKA dependencies
        InputStream stream = FastRandomForestBenchmark.class.getResourceAsStream("logging.properties");
        try {
            LogManager.getLogManager().readConfiguration(stream);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static final int TRAINING_SIZE = 1_000_000;
    private static final int TESTING_SIZE = 5_000_000;

    private Instances testData;
    private FastRandomForest trainedModel;

    @Setup
    public void setup() throws Exception {
        trainedModel = new FastRandomForest();
        trainedModel.setNumTrees(10);
        trainedModel.buildClassifier(generateInstances(TRAINING_SIZE));

        testData = generateInstances(TESTING_SIZE);
    }

    @Benchmark
    public void run() throws Exception {
        List<Double> results = new ArrayList<>(testData.size());

        for (Instance instance : testData) {
            double result = trainedModel.classifyInstance(instance);
            results.add(result);
        }

        if (results.size() != testData.size()) {
            throw new Exception();
        }
    }

    protected abstract Instances generateInstances(int n);

}
