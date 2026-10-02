package com.llm.app;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class Trainer {
  private int batchSize;
  private double learningRate;
  private double warmupRate;
  private int trainLines;
  private String splitRegex;

  public Trainer(int BATCH_SIZE, double LEARNING_RATE, double WARM_UP_RATE, int TRAIN_LINES, String splitRegex) {
    this.batchSize = BATCH_SIZE;
    this.learningRate = LEARNING_RATE;
    this.warmupRate = WARM_UP_RATE;
    this.trainLines = TRAIN_LINES;
    this.splitRegex = splitRegex;
  }

  public double train() throws FileNotFoundException {
    System.out.println("Starting Training...");
    Map<String, String> trainingLookup = createTrainingDataset();
    double loss = -1.0;
    return loss;
  }

  // todo: Implement batch size

  public Map<String, String> createTrainingDataset() throws FileNotFoundException {
    Map<String, String> trainingDataLookup = new LinkedHashMap<>();
    String DATA_FILE_PATH = "./dataset/speeches.csv";
    String line = "";
    AtomicInteger count = new AtomicInteger(0);
    try (BufferedReader reader = new BufferedReader(new FileReader(DATA_FILE_PATH))) {
      while ((line = reader.readLine()) != null) {
        Map<String, String> trainingDataLookupPointer = new LinkedHashMap<>();
        List<String> vocabWords = List.of(line.split(splitRegex));
        for (int i = 0; i < vocabWords.size() - 1; i++) {
          if(trainingDataLookupPointer.keySet().toArray().length == 0) {
            trainingDataLookupPointer.put(vocabWords.get(i), vocabWords.get(i) + " " + vocabWords.get(i + 1));
          } else {
            String key = trainingDataLookupPointer.get(trainingDataLookupPointer.keySet().stream().toList().get(i-1))
                + " " + vocabWords.get(i);
            String value = key + " " + vocabWords.get(i + 1);
            trainingDataLookupPointer.put(key, value);
          }
        }
        count.getAndIncrement();
        if(count.intValue() == trainLines) {
          break;
        }
        trainingDataLookup.putAll(trainingDataLookupPointer);
      }

    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return trainingDataLookup;
  }
}
