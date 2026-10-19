package com.llm.app;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import jdk.jshell.execution.Util;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class SoftmaxTokenPredictor {
  private final Map<String, Integer> vocabulary;
  private int seed;
  private final double sigma;
  private RealMatrix linearHead;
  private RealMatrix meanEmbeddings;

  public SoftmaxTokenPredictor(Map<String, Integer> vocabulary, int seed, double sigma) {
    this.vocabulary = vocabulary;
    this.seed = seed;
    this.sigma = sigma;
  }

  public RealMatrix generateSoftmaxResult(RealMatrix meanEmbeddings, double temperature) throws IOException {
    this.meanEmbeddings = meanEmbeddings;
    String predictionLinearHeadFilePath = "./model/transformer/prediction_weights/linear_head.txt";
    this.linearHead = Utils.buildMatrixIfNotExists(seed, sigma, meanEmbeddings.getColumnDimension(), vocabulary.size(), predictionLinearHeadFilePath);
    this.seed += 1;
    
    RealMatrix headResult = Utils.multipleMatrices(meanEmbeddings, linearHead);
    System.out.println("calculated head result: ");
    Utils.printMatrix((Array2DRowRealMatrix) headResult, true);
    RealMatrix softmaxResult = Utils.calculateSoftmax(headResult, temperature);
    System.out.println("calculated softmax result: ");
    Utils.printMatrix((Array2DRowRealMatrix) softmaxResult, true);
    return softmaxResult;
  }

  public String predictNextToken(RealMatrix softmaxResult) {
    String nextToken = "";

    List<Double> consideredOutputs = new ArrayList<>();
    for (int column = 0; column < softmaxResult.getColumnDimension(); column++) {
      double softmaxEntry = softmaxResult.getEntry(softmaxResult.getRowDimension() - 1, column);
      softmaxEntry += column;
      consideredOutputs.add(softmaxEntry);
    }

    Random random = new Random();
    double predictionRandomValue =  random.nextDouble();
    double cum = 0.0;
    int nextTokenIndex = 0;
    for (int i = 0; i < consideredOutputs.size(); i++) {
      double entry = consideredOutputs.get(i);
      if (entry - i <= 0.0) {
        continue;
      }
      int index = 0;
      for (int j = 0; j < softmaxResult.getColumnDimension(); j++) {
        if((entry - j - 1) <= 0)  {
          index = j;
          entry = entry - j;
          break;
        }
      }
      cum += entry;
      if(predictionRandomValue < cum) {
        nextTokenIndex = index;
        break;
      }
    }
    System.out.println("Predicted next token index: " + nextTokenIndex);
    for (int i = 0; i < vocabulary.keySet().size(); i++) {
      if(vocabulary.get(vocabulary.keySet().toArray()[i]) == nextTokenIndex) {
        nextToken = vocabulary.keySet().toArray()[i].toString();
        break;
      }
    }
    return nextToken;
  }

  public int getSeed() {
    return seed;
  }

  public RealMatrix backward(RealMatrix softmaxResult, RealMatrix oneHotVector, double learningRate, double trainTemperature) throws IOException {
    if(trainTemperature <= 0) {
      throw new IllegalArgumentException("temperature must be > 0 during tarining");
    }

    // calculate gradient
    RealMatrix X = this.meanEmbeddings.getRowMatrix(this.meanEmbeddings.getRowDimension() - 1);
    RealMatrix Z = softmaxResult;

    RealMatrix dZ = Z.subtract(oneHotVector).scalarMultiply(1.0 / trainTemperature);
    RealMatrix dW = Utils.multipleMatrices(X.transpose(), dZ);
    RealMatrix dX = Utils.multipleMatrices(dZ, this.linearHead.transpose());

    // update weight matrix
    this.linearHead = this.linearHead.subtract(dW.scalarMultiply(learningRate));
    Utils.overrideFileMatrix("./model/transformer/prediction_weights/linear_head.txt", this.linearHead);
    return dX;
  }
}
