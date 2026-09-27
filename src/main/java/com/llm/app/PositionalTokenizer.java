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
  private int tokenCount;
  private int dimensions;

  public PositionalTokenizer(int tokenCount, int dimensions) {
    this.tokenCount = tokenCount;
    this.dimensions = dimensions;
    System.out.println("---------POSITIONAL TOKENIZER LAYER---------");
  }

  private final String DATA_FILE_PATH = "./dataset/speeches.csv";
  private final String EMBEDDING_FILE_PATH = "./result/embedding_lookup/embedding_matrix.txt";

  public RealMatrix embeddSentence(String sentence) throws IOException {
    Map<String, Integer> vocabulary = buildVocabulary();
    RealMatrix embeddingMatrix = Utils.buildMatrixIfNotExists(vocabulary.keySet().size(), dimensions, EMBEDDING_FILE_PATH);
    RealMatrix tokenPosEmbeddings = new Array2DRowRealMatrix(vocabulary.keySet().size(), dimensions);
    List<String> wordList = List.of(sentence.split(" |(\\n)"));

    AtomicInteger positionEncoding = new AtomicInteger();
    AtomicInteger counter = new AtomicInteger(0);
    wordList.forEach(word -> {

      // tokenize
      double[] tokenEmbedding = wordEncode(positionEncoding.get(), word, vocabulary, embeddingMatrix);
      System.out.println("word: " + word);
      System.out.print("embedding: ");
      System.out.print("[");

      // generate positional embedding
      for (int i = 0; i < dimensions; i++) {
        System.out.print(tokenEmbedding[i] + " ");
        tokenPosEmbeddings.setEntry(counter.get(), i, tokenEmbedding[i]);
      }
      System.out.println("]");
      System.out.println();


      positionEncoding.getAndIncrement();
      counter.getAndIncrement();
    });
    return tokenPosEmbeddings;
  }

  public double[] wordEncode(int position, String word, Map<String, Integer> vocabulary, RealMatrix embeddingMatrix) {
    double[] wordEmbeddingVector;
    double[] positionalEmbedding = new double[dimensions];
    if(!(vocabulary.get(word) == null)) {
      int tokenNumber = vocabulary.get(word);
      wordEmbeddingVector = embeddingMatrix.getRow(tokenNumber);
    } else {
      int tokenNumber =  vocabulary.get("NULL");
      System.out.println("Word: " + word + " not known.");
      wordEmbeddingVector = embeddingMatrix.getRow(tokenNumber);
    }
    for (int i = 0; i < positionalEmbedding.length; i++) {
      if(i%2 == 0) {
        // even indices use sinusoidal pos embeddings
        positionalEmbedding[i] = Math.sin(position / Math.pow(10000.0, (2*i) / dimensions));
      } else {
        // odd indices use cosinoidal pos embeddings
        positionalEmbedding[i] = Math.cos(position / Math.pow(10000.0, (2*i) / dimensions));
      }
    }
    return addVectors(wordEmbeddingVector, positionalEmbedding);
  }

  public Map<String, Integer > buildVocabulary() {
    System.out.println("Building Vocabulary..");
    Map<String, Integer> vocabulary = new HashMap<>();
    String line = "";
    int vocabLimit = this.tokenCount; // max value is 25605402
    AtomicInteger count = new AtomicInteger(0);
    vocabulary.put("NULL", 0);
    AtomicInteger wordIndex = new AtomicInteger(1);
    try (BufferedReader reader = new BufferedReader(new FileReader(DATA_FILE_PATH))) {
      while ((line = reader.readLine()) != null) {
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

  public double[] addVectors(double[] vec1, double[] vec2) {
    if(vec1.length != vec2.length) {
      throw new IllegalArgumentException("vectors of different lengths can not be added");
    }
    double[] result = new double[vec1.length];
    for (int i = 0; i < vec1.length; i++) {
      result[i] = vec1[i] + vec2[i];
    }
    return result;
  }
}
