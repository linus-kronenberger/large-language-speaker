package com.llm.app;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class PositionalTokenizer {
  public PositionalTokenizer() {
    System.out.println("---------POSITIONAL TOKENIZER LAYER---------");
  }

  private final int EMBEDDING_DIMENSIONS = 64;
  private final String DATA_FILE_PATH = "./dataset/speeches.csv";

  public List<List<Double>> embeddSentence(String sentence) {
    Map<String, Integer> vocabulary = buildVocabulary();
    RealMatrix embeddingMatrix = buildEmbeddingMatrix(vocabulary.keySet().size(), EMBEDDING_DIMENSIONS,100);
    List<List<Double>> tokenPosEmbeddings = new ArrayList<>(List.of());
    List<String> wordList = List.of(sentence.split(" |(\\n)"));

    AtomicInteger positionEncoding = new AtomicInteger();
    wordList.forEach(word -> {

      // tokenize
      double[] tokenEmbedding = wordEncode(positionEncoding.get(), word, vocabulary, embeddingMatrix);
      System.out.println("word: " + word);
      System.out.print("embedding: ");
      System.out.print("[");

      // generate positional embedding
      for (int i = 0; i < EMBEDDING_DIMENSIONS; i++) {
        System.out.print(tokenEmbedding[i] + " ");
      }
      System.out.println("]");
      System.out.println();

      Double[] doubleObjectVector = Arrays.stream(tokenEmbedding)
          .mapToObj(Double::valueOf)
          .toArray(Double[]::new);
      tokenPosEmbeddings.add(List.of(doubleObjectVector));
      positionEncoding.getAndIncrement();
    });
    return tokenPosEmbeddings;
  }

  public double[] wordEncode(int position, String word, Map<String, Integer> vocabulary, RealMatrix embeddingMatrix) {
    double[] wordEmbeddingVector;
    double[] positionalEmbedding = new double[EMBEDDING_DIMENSIONS];
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
        positionalEmbedding[i] = Math.sin(position / Math.pow(10000.0, (2*i) / EMBEDDING_DIMENSIONS));
      } else {
        // odd indices use cosinoidal pos embeddings
        positionalEmbedding[i] = Math.cos(position / Math.pow(10000.0, (2*i) / EMBEDDING_DIMENSIONS));
      }
    }
    return addVectors(wordEmbeddingVector, positionalEmbedding);
  }

  public Map<String, Integer > buildVocabulary() {
    System.out.println("Building Vocabulary..");
    Map<String, Integer> vocabulary = new HashMap<>();
    String line = "";
    int vocabLimit = 100; // max value is 25605402
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
        // embeddingMatrix.setEntry(i, 0, i);
        for (int j = 0; j < dimensions; j++) {
          // create random object
          Random r = new Random();
          // generating integer
          double n = r.nextGaussian();
          // Printing the random Number
          System.out.println("The next Gaussian value generated is: " + n);
          embeddingMatrix.setEntry(i, j, n);
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
