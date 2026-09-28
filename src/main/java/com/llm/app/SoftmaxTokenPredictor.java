package com.llm.app;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class SoftmaxTokenPredictor {
  private Map<String, Integer> vocabulary;
  private int seed;
  private double sigma;

  public SoftmaxTokenPredictor(Map<String, Integer> vocabulary, int seed, double sigma, int attentionHeads) {
    this.vocabulary = vocabulary;
    this.seed = seed + attentionHeads * 3 + 3;
    this.sigma = sigma;
  }
  public String predictNextToken(RealMatrix meanEmbeddings, double temperature) throws IOException {
    String nextToken = "";
    String predictionLinearHeadFilePath = "./result/prediction_weights/linear_head.txt";
    RealMatrix linearHead = Utils.buildMatrixIfNotExists(seed, sigma, meanEmbeddings.getColumnDimension(), vocabulary.size(), predictionLinearHeadFilePath);
    
    RealMatrix headResult = Utils.multipleMatrices(meanEmbeddings, linearHead);
    System.out.println("calculated head result: ");
    Utils.printMatrix((Array2DRowRealMatrix) headResult, true);
    RealMatrix softmaxResult = Utils.calculateSoftmax(headResult, temperature);
    System.out.println("calculated softmax result: ");
    Utils.printMatrix((Array2DRowRealMatrix) softmaxResult, true);
    List<Double> consideredOutputs = new ArrayList<>();
    for (int column = 0; column < softmaxResult.getColumnDimension(); column++) {
      double softmaxEntry = softmaxResult.getEntry(softmaxResult.getRowDimension() - 1, column);
      softmaxEntry += column;
      consideredOutputs.add(softmaxEntry);
    }

    double predictionRandomValue = Math.random();
    double cum = 0.0;
    int nextTokenIndex = 0;
    for (int i = 0; i < consideredOutputs.size(); i++) {
      double entry = consideredOutputs.get(i);
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
}
