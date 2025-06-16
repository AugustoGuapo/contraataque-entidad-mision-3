package org.example;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

public class Evaluacion {
    public static void main(String[] args) {
        /*try {
            // Cargar el modelo previamente entrenado
            MultiLayerNetwork model = ModelLoader.loadModel("src/main/resources/saved_models/modelo2.zip");

            // Evaluar el modelo con el conjunto de prueba
            Evaluation eval = new Evaluation(2); // 2 clases: original y modificada
            while (testIter.hasNext()) {
                DataSet testData = testIter.next();
                INDArray output = model.output(testData.getFeatures());
                eval.eval(testData.getLabels(), output);
            }

            // Imprimir los resultados de la evaluación
            System.out.println(eval.stats());
        } catch (IOException e) {
            e.printStackTrace();*/
    }
}
