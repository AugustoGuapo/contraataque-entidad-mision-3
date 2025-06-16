package org.example;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;

public class ModelLoader {
    public static MultiLayerNetwork loadModel(String modelPath) throws IOException {
        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            throw new IOException("Model file not found: " + modelPath);
        }
        return ModelSerializer.restoreMultiLayerNetwork(modelFile);
    }
}
