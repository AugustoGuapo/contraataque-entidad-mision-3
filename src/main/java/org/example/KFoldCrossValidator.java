package org.example;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class KFoldCrossValidator {

    /*public static MultiLayerNetwork runKFold(String imageDirPath, int numFolds, int height, int width, int channels, int batchSize, int outputNum, MultiLayerConfiguration config) throws IOException {
        File imageDir = new File(imageDirPath);
        File[] imageFiles = imageDir.listFiles();
        File[] imageFilesOriginals = imageDir.listFiles((dir, name) -> dir.getName().equalsIgnoreCase("originals"));
        File[] imageFilesAugmented = imageDir.listFiles((dir, name) -> dir.getName().equalsIgnoreCase("augmented"));


        // Shuffle the list
        List<File> allFiles = Arrays.asList(imageFiles);
        Collections.shuffle(allFiles, new Random(42));

        int foldSize = allFiles.size() / numFolds;
        double bestScore = Double.MAX_VALUE;
        MultiLayerNetwork bestModel = null;

        for (int fold = 0; fold < numFolds; fold++) {
            System.out.println("\n========== Fold " + (fold + 1) + "/" + numFolds + " ==========");

            List<File> testFiles = allFiles.subList(fold * foldSize, (fold + 1) * foldSize);
            List<File> trainFiles = new ArrayList<>(allFiles);
            trainFiles.removeAll(testFiles);

            // Crear carpetas temporales
            File tempTrainDir = Files.createTempDirectory("train_fold_" + fold).toFile();
            File tempTestDir = Files.createTempDirectory("test_fold_" + fold).toFile();

            copyFilesToTempDir(trainFiles, tempTrainDir);
            copyFilesToTempDir(testFiles, tempTestDir);

            // Train split
            FileSplit trainSplit = new FileSplit(tempTrainDir, NativeImageLoader.ALLOWED_FORMATS, new Random());
            ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, new ParentPathLabelGenerator());
            trainRR.initialize(trainSplit);
            DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

            // Test split
            FileSplit testSplit = new FileSplit(tempTestDir, NativeImageLoader.ALLOWED_FORMATS, new Random());
            ImageRecordReader testRR = new ImageRecordReader(height, width, channels, new ParentPathLabelGenerator());
            testRR.initialize(testSplit);
            DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);

            // Normalizar
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            scaler.fit(trainIter);
            trainIter.setPreProcessor(scaler);
            testIter.setPreProcessor(scaler);

            MultiLayerNetwork model = new MultiLayerNetwork(config);
            model.init();
            model.setListeners(new ScoreIterationListener(10));

            EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                    .epochTerminationConditions(new org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition(100))
                    .scoreCalculator(new org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator(testIter, true))
                    .evaluateEveryNEpochs(1)
                    .build();

            EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, model, trainIter);
            EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

            MultiLayerNetwork bestFoldModel = result.getBestModel();
            double score = result.getBestModelScore();

            Evaluation eval = bestFoldModel.evaluate(testIter);
            System.out.println(eval.stats());

            if (score < bestScore) {
                bestScore = score;
                bestModel = bestFoldModel;
            }

            // Cleanup
            deleteDirectory(tempTrainDir);
            deleteDirectory(tempTestDir);
        }

        return bestModel;
    }

    private static void copyFilesToTempDir(List<File> files, File targetDir) throws IOException {
        for (File f : files) {
            Path targetPath = new File(targetDir, f.getName()).toPath();
            Files.copy(f.toPath(), targetPath, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void deleteDirectory(File dir) {
        if (dir.isDirectory()) {
            for (File file : dir.listFiles()) {
                deleteDirectory(file);
            }
        }
        dir.delete();
    }*/
}
