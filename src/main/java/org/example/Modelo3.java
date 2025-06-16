package org.example;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class Modelo3 {
    public static void main(String[] args) throws IOException {
        /*int height = 128; // Altura de las imágenes
        int width = 128; // Ancho de las imágenes
        int channels = 1; // 1 para imágenes en escala de grises, 3 para RGB
        int outputNum = 2; // original / modificada
        int batchSize = 64; // Tamaño del lote
        int epochs = 100; // Número de épocas para entrenar el modelo

        Random rng = new Random(123); // Semilla fija para reproducibilidad
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


        MultiLayerConfiguration config = getConfigForThirdModel(channels, outputNum, height, width);


// Inicializar el modelo con la configuración
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

// Configuración de EarlyStopping
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(epochs)) // máx. 100 epochs
                .evaluateEveryNEpochs(1) // evaluar cada epoch
                .scoreCalculator(new DataSetLossCalculator(testIter, true)) // usa test set como validación
                .modelSaver(new InMemoryModelSaver<>()) // guarda modelo en memoria
                .build();

// Entrenar con EarlyStopping
        long startTime = System.currentTimeMillis();
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, model, trainIter);
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

// Obtener el mejor modelo entrenado
        MultiLayerNetwork bestModel = result.getBestModel();

// Evaluar usando el test set
        Evaluation eval = bestModel.evaluate(testIter);
        System.out.println("Tiempo total de entrenamiento: " + (System.currentTimeMillis() - startTime) / 1000 + " segundos");
        System.out.println(eval.stats());
        // Save to json evaluation
        File evalFile = new File("src/main/resources/saved_models/model3/evaluation.json");
        evalFile.getParentFile().mkdirs();
        org.apache.commons.io.FileUtils.writeStringToFile(evalFile, eval.toJson(), java.nio.charset.StandardCharsets.UTF_8);

// Guardar el modelo entrenado
        File modeloGuardado = new File("src/main/resources/saved_models/model3/model.zip");
        modeloGuardado.getParentFile().mkdirs();
        ModelSerializer.writeModel(bestModel, modeloGuardado, true);
    }

    private static MultiLayerConfiguration getConfigForThirdModel(int channels, int outputNum, int height, int width) {
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
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels)) // importante para calcular tamaños internos
                .build();*/
    }
}
