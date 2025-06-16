package org.example;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

public class Main {
    public static void main(String[] args) throws IOException {
        int height = 80; // Altura de las imágenes
        int width = 70; // Ancho de las imágenes
        int channels = 1; // 1 para imágenes en escala de grises, 3 para RGB
        int outputNum = 2; // original / modificada
        int batchSize = 32; // Tamaño del lote
        int epochs = 100; // Número de épocas para entrenar el modelo

        Random rng = new Random(123); // Semilla fija para reproducibilidad
        String datasetPath = "src/main/resources/dataset";
        File testDatasetDir = new File("src/main/resources/dataset_final/test");
        File datasetDir = new File("src/main/resources/dataset_final/train");
// 1. Crear el FileSplit
        FileSplit fileSplitTrain = new FileSplit(datasetDir, NativeImageLoader.ALLOWED_FORMATS, rng);
        FileSplit fileSplitTest = new FileSplit(testDatasetDir, NativeImageLoader.ALLOWED_FORMATS, rng);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
        trainRR.initialize(fileSplitTrain);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(fileSplitTest);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        // Calcular pesos de clase para manejar el desbalanceo
        double[] classWeights = calcularPesosClase(trainIter, outputNum);
        System.out.println("Pesos de clase: " + Arrays.toString(classWeights));


        MultiLayerConfiguration config = getConfigForFourthModel(channels, classWeights, outputNum, height, width);
        long startTime = System.currentTimeMillis();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.fit(trainIter, epochs);

        Evaluation eval = model.evaluate(testIter);
        System.out.println("Tiempo total de entrenamiento: " + (System.currentTimeMillis() - startTime) / 1000 + " segundos");
        System.out.println(eval.stats());
        // Save to json evaluation
        File evalFile = new File("src/main/resources/saved_models/model4/evaluation.json");
        evalFile.getParentFile().mkdirs();
        org.apache.commons.io.FileUtils.writeStringToFile(evalFile, eval.toJson(), java.nio.charset.StandardCharsets.UTF_8);

// Guardar el modelo entrenado
        File modeloGuardado = new File("src/main/resources/saved_models/model4/model.zip");
        modeloGuardado.getParentFile().mkdirs();
        ModelSerializer.writeModel(model, modeloGuardado, true);
    }

    /*private static MultiLayerConfiguration getConfigForFirstModel(int channels, double[] classWeights, int outputNum, int height, int width) {
        return new NeuralNetConfiguration.Builder()
                .updater(new org.nd4j.linalg.learning.config.Adam(0.001))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .nOut(32)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(64)
                        .build())
                .layer(new OutputLayer.Builder(new LossMCXENT(Nd4j.create(classWeights)))
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
    }

    private static MultiLayerConfiguration getConfigForSecondModel(int channels, double[] classWeights, int outputNum, int height, int width) {
        return new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)  // solo en la primera capa
                        .nOut(32)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(64)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(new LossMCXENT(Nd4j.create(new double[] {classWeights[0] * 1, classWeights[1] * 1.5})))
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels)) // importante para calcular tamaños internos
                .build();
    }*/

    private static MultiLayerConfiguration getConfigForThirdModel(int channels, double[] classWeights, int outputNum, int height, int width) {
        return new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)  // solo en la primera capa
                        .nOut(32)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(64)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(128)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(new LossMCXENT(Nd4j.create(new double[] {classWeights[0] * 1, classWeights[1] * 1.25})))
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels)) // importante para calcular tamaños internos
                .build();
    }

    private static MultiLayerConfiguration getConfigForFourthModel(int channels, double[] classWeights, int outputNum, int height, int width) {
        return new NeuralNetConfiguration.Builder()
                .updater(new Adam(new MapSchedule(ScheduleType.EPOCH,
                        Map.of(0, 0.001, 30, 0.0005, 60, 0.0001))))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)  // solo en la primera capa
                        .nOut(32)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(64)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(128)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DropoutLayer(0.25))
                .layer(new OutputLayer.Builder(new LossMCXENT(Nd4j.create(new double[] {classWeights[0] * 1, classWeights[1] * 1.375})))
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels)) // importante para calcular tamaños internos
                .build();
    }

    //VGG inspired model
    private static MultiLayerConfiguration getConfigForFifthModel(int channels, double[] classWeights, int outputNum, int height, int width) {
        return new NeuralNetConfiguration.Builder()
                .updater(new Adam(new MapSchedule(ScheduleType.EPOCH,
                        Map.of(0, 0.001, 30, 0.0005, 60, 0.0001))))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .nOut(32)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(64)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(128)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(256)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DropoutLayer(0.4))
                .layer(new OutputLayer.Builder(
                        new LossMCXENT(Nd4j.create(new double[] {
                                classWeights[0],
                                classWeights[1] * 1.375
                        })))
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
    }

    public static double[] calcularPesosClase(DataSetIterator iter, int numClases) {
        long[] classCounts = new long[numClases];
        long totalSamples = 0;

        while (iter.hasNext()) {
            INDArray labels = iter.next().getLabels();
            for (int i = 0; i < labels.rows(); i++) {
                int label = Nd4j.argMax(labels.getRow(i), 0).getInt(0);
                classCounts[label]++;
                totalSamples++;
            }
        }
        iter.reset(); // ⚠️ ¡Importante!

        double[] classWeights = new double[numClases];
        for (int i = 0; i < numClases; i++) {
            classWeights[i] = (double) totalSamples / (numClases * classCounts[i]);
        }

        return classWeights;
    }
}