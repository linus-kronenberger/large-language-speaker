package com.llm.app;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Trainer {
  private int batchSize;
  private double learningRate;
  private double warmupRate;
  private int trainLines;
  private String splitRegex;
  private int givenSentenceLength = 5;
  private int minWordCount = 20;

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

  public Map<String, String> createTrainingDataset() {
    Map<String, String> trainingDataLookup = new LinkedHashMap<>();
    String DATA_FILE_PATH = "./dataset/speeches.csv";
    String line = "";
    int countLines = 0;
    try (BufferedReader reader = new BufferedReader(new FileReader(DATA_FILE_PATH))) {
      reader.readLine();
      reader.readLine();
      while ((line = reader.readLine()) != null) {

        Map<String, String> trainingDataLookupPointer = new LinkedHashMap<>();
        List<String> vocabWords = List.of(line.split(splitRegex));

        if(vocabWords.size() < minWordCount) {
          continue;
        }

        // First add the given sentence
        if(vocabWords.size() == 1) {
          continue;
        }
        else if(vocabWords.size() > givenSentenceLength) {
          String concatSentence = "";
          for (int i = 0; i < givenSentenceLength; i++) {
            if(i == givenSentenceLength - 1) {
              concatSentence += vocabWords.get(i);
            } else {
              concatSentence += vocabWords.get(i) + " ";
            }

          }

          String firstKey = concatSentence;
          String firstValue = concatSentence + " " + vocabWords.get(givenSentenceLength);

          trainingDataLookupPointer.put(firstKey, firstValue);
        }
        else {
          // String firstKey = vocabWords.get(0);
          // String firstValue = firstKey + " " + vocabWords.get(1);
          // trainingDataLookupPointer.put(firstKey, firstValue);
          continue;
        }

        String currentKey = trainingDataLookupPointer.values().iterator().next();
        int contextSize = (vocabWords.size() > givenSentenceLength) ? givenSentenceLength : 1;
        for (int i = contextSize + 1; i < vocabWords.size(); i++) {
          String value = currentKey + " " + vocabWords.get(i);
          trainingDataLookupPointer.put(currentKey, value);
          currentKey = value;
        }
        countLines++;
        trainingDataLookup.putAll(trainingDataLookupPointer);
        if(countLines == trainLines) {
          break;
        }
      }

    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    System.out.println(trainingDataLookup);
    return trainingDataLookup;
  }
}
