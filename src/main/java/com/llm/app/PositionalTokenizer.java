package com.llm.app;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
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
    List<List<Float>> tokenPosEmbeddings = new ArrayList<>(List.of());
    List<String> wordList = List.of(sentence.split(" |(\\n)"));
    wordList.forEach(word -> {

      // tokenize
      int token = wordEncode(word, vocabulary);
      List<Float> tokenEmbedding = new ArrayList<>(List.of());
      float tokenFloat = (float) token;
      System.out.print("[");

      // generate positional embedding
      for (int i = 0; i < EMBEDDING_DIMENSIONS; i++) {
        tokenEmbedding.add(tokenFloat);
        System.out.print(tokenFloat + " ");
      }
      System.out.println("]");
      System.out.println();
      tokenPosEmbeddings.add(tokenEmbedding);
    });
    RealMatrix embeddingMatrix = buildEmbeddingMatrix(vocabulary.keySet().size(), EMBEDDING_DIMENSIONS,100);
    return tokenPosEmbeddings;
  }

  public int wordEncode(String word, Map<String, Integer> vocabulary) {
    if(!(vocabulary.get(word) == null)) {
      return vocabulary.get(word);
    } else {
      return vocabulary.get("NULL");
    }
  }

  public Map<String, Integer > buildVocabulary() {
    System.out.println("Building Vocabulary..");
    Map<String, Integer> vocabulary = new HashMap<>();
    String line = "";
    int vocabLimit = 10;
    int count = 0;
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
        count ++;
        if(count == vocabLimit) {
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
    RealMatrix embeddingMatrix = new Array2DRowRealMatrix(tokenCount, dimensions);
    boolean matrixFinished = false;
    while (matrixFinished == false) {
      for(int i=0; i<tokenCount; i++) {
        for (int j = 0; j < dimensions; j++) {
          int matrixEntry = (int) (Math.random() * maxValue) + 1;
          embeddingMatrix.setEntry(i, j, matrixEntry);
        }
      }
      matrixFinished = true;
    }
    System.out.printf("Embedding Matrix: " + embeddingMatrix.toString());
    return embeddingMatrix;
    // fallback: take matrix from ./model path if available
  }
}
