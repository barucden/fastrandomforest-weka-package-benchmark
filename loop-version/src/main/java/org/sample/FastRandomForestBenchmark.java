/*
 * Copyright (c) 2014, Oracle America, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 *  * Neither the name of Oracle nor the names of its contributors may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

package org.sample;

import hr.irb.fastRandomForest.FastRandomForest;
import org.openjdk.jmh.annotations.*;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.logging.LogManager;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 1)
@Measurement(iterations = 2)
@Fork(value = 1)
public class FastRandomForestBenchmark {

    static {
        //disable annoying logs from some WEKA dependencies
        InputStream stream = FastRandomForestBenchmark.class.getResourceAsStream("logging.properties");
        try {
            LogManager.getLogManager().readConfiguration(stream);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static final int TRAINING_SIZE = 10_000;
    private static final int TESTING_SIZE = 3_000_000;

    private Instances testData;
    private FastRandomForest trainedModel;

    @Setup()
    public void setup() throws Exception {
        trainedModel = new FastRandomForest();
        trainedModel.setNumTrees(10);
        trainedModel.buildClassifier(generateInstances(TRAINING_SIZE));

        testData = generateInstances(TESTING_SIZE);
    }

    @Benchmark
    public void testMethod() throws Exception {
        List<Double> results = new ArrayList<>(testData.size());

        for (Instance instance : testData) {
            double result = trainedModel.classifyInstance(instance);
            results.add(result);
        }

        if (results.size() != testData.size()) {
            throw new Exception();
        }
    }

    private static Instances generateInstances(int n) {
        List<Attribute> attributes = Arrays.asList(new Attribute("attr1"),
                                                   new Attribute("attr2"),
                                                   new Attribute("attr3"),
                                                   new Attribute("attr4"),
                                                   new Attribute("attr5"));
        List<String> classes = Arrays.asList("class1", "class2");
        ArrayList<Attribute> attributesAndClass = new ArrayList<>(attributes);
        attributesAndClass.add(new Attribute("class", classes));

        Instances instances = new Instances("dataset", attributesAndClass, n);
        instances.setClassIndex(attributesAndClass.size() - 1);

        ThreadLocalRandom random = ThreadLocalRandom.current();
        for (int i = 0; i < n; i++) {
            Instance instance = new DenseInstance(attributesAndClass.size());
            instance.setDataset(instances);

            for (int attr = 0; attr < attributes.size(); attr++) {
                double value = random.nextDouble(-10, 10);
                instance.setValue(attr, value);
            }

            instance.setClassValue(classes.get(random.nextInt(classes.size())));
            instances.add(instance);
        }
        return instances;
    }

}
