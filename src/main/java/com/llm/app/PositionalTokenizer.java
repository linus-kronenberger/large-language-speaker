package com.llm.app;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class PositionalTokenizer {

  private final int EMBEDDING_DIMENSIONS = 64;
  private final String DATA_FILE_PATH = "./dataset/speeches.csv";

  public List<List<Float>> embeddSentence(String sentence) {
    Map<String, Integer> vocabulary = buildVocabulary();
    RealMatrix embeddingMatrix = buildEmbeddingMatrix(vocabulary.keySet().size(), EMBEDDING_DIMENSIONS,100);
    List<List<Float>> tokenPosEmbeddings = new ArrayList<>(List.of());
    List<String> wordList = List.of(sentence.split(" |(\\n)"));

    AtomicInteger positionEncoding = new AtomicInteger();

    wordList.forEach(word -> {

      // tokenize
      double[] tokenEmbedding = wordEncode(word, vocabulary, embeddingMatrix);
      System.out.println("word: " + word);
      System.out.print("embedding: ");
      System.out.print("[");

      // generate positional embedding
      for (int i = 0; i < EMBEDDING_DIMENSIONS; i++) {
        System.out.print(tokenEmbedding[i] + " ");
      }
      System.out.println("]");
      System.out.println();
      // tokenPosEmbeddings.add(tokenEmbedding);
      positionEncoding.getAndIncrement();
    });
    return tokenPosEmbeddings;
  }

  public double[] wordEncode(String word, Map<String, Integer> vocabulary, RealMatrix embeddingMatrix) {
    if(!(vocabulary.get(word) == null)) {
      int tokenNumber = vocabulary.get(word);
      return embeddingMatrix.getRow(tokenNumber);
    } else {
      int tokenNumber =  vocabulary.get("NULL");
      return embeddingMatrix.getRow(tokenNumber);
    }
  }

  public Map<String, Integer > buildVocabulary() {
    System.out.println("Building Vocabulary..");
    Map<String, Integer> vocabulary = new HashMap<>();
    String line = "";
    int vocabLimit = 25605402;
    AtomicInteger count = new AtomicInteger(0);
    vocabulary.put("NULL", 0);
    AtomicInteger wordIndex = new AtomicInteger(1);
    try (BufferedReader br = new BufferedReader(new FileReader(DATA_FILE_PATH))) {
      while ((line = br.readLine()) != null) {
        System.out.println(line);
        List<String> vocabWords = List.of(line.split(" "));
        vocabWords.forEach(word -> {
          if(!vocabulary.containsKey(word)) {
            vocabulary.put(word, wordIndex.get());
            System.out.println("Added token to vocabulary: " + wordIndex + "   " + word);
            wordIndex.getAndIncrement();
          }
        });
        count.getAndIncrement();
        if(count.intValue() == vocabLimit) {
          break;
        }
      }
    } catch (IOException e) {
      e.printStackTrace();
    }

    return vocabulary;
  }

  public RealMatrix buildEmbeddingMatrix(int tokenCount, int dimensions, int maxValue) {
    // loop for building embedding matrix from scratch
    System.out.println("Building Embedding Matrix..");
    Array2DRowRealMatrix embeddingMatrix = new Array2DRowRealMatrix(tokenCount, dimensions);
    boolean matrixFinished = false;
    while (matrixFinished == false) {
      for(int i=0; i<tokenCount; i++) {
        embeddingMatrix.setEntry(i, 0, i);
        for (int j = 1; j < dimensions; j++) {
          int matrixEntry = (int) (Math.random() * maxValue) + 1;
          embeddingMatrix.setEntry(i, j, matrixEntry);
        }
      }
      matrixFinished = true;
    }
    printMatrix(embeddingMatrix);
    return embeddingMatrix;
    // fallback: take matrix from ./model path if available
  }

  public void printMatrix(Array2DRowRealMatrix matrix) {
    System.out.println("Embedding rows: " + matrix.getRowDimension());
    System.out.println("Embedding columns: " + matrix.getColumnDimension());
    for (int i = 0; i < matrix.getRowDimension(); i++) {
      System.out.println(Arrays.toString(matrix.getRow(i)));
    }
  }
}
