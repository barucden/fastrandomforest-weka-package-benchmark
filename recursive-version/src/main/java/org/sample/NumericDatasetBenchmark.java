package org.sample;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class NumericDatasetBenchmark extends FastRandomForestBenchmark {

    protected Instances generateInstances(int n) {
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
                double value = random.nextDouble(-50, 50);
                instance.setValue(attr, value);
            }

            instance.setClassValue(classes.get(random.nextInt(classes.size())));
            instances.add(instance);
        }
        return instances;
    }

}
