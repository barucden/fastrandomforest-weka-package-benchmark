package org.sample;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class NominalDatasetBenchmark extends FastRandomForestBenchmark {

    private Instances generateInstances(int n) {
        List<Attribute> attributes = Arrays.asList(new Attribute("attr1",
                                                                 Arrays.asList("v1", "v2", "v3", "v4", "v5", "v6", "v7")),
                                                   new Attribute("attr2",
                                                                 Arrays.asList("v1", "v2", "v3", "v4", "v5", "v6")),
                                                   new Attribute("attr3",
                                                                 Arrays.asList("v1", "v2", "v3", "v4", "v5")),
                                                   new Attribute("attr4",
                                                                 Arrays.asList("v1", "v2", "v3", "v4", "v5", "v6", "v7")),
                                                   new Attribute("attr5",
                                                                 Arrays.asList("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8")));
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
                Attribute attribute = attributes.get(attr);
                String value = attribute.value(random.nextInt(attribute.numValues()));
                instance.setValue(attr, value);
            }

            instance.setClassValue(classes.get(random.nextInt(classes.size())));
            instances.add(instance);
        }
        return instances;
    }

    @Override
    protected Instances getTrainingData() {
        return generateInstances(1_000_000);
    }

    @Override
    protected Instances getTestingData() {
        return generateInstances(5_000_000);
    }

}
