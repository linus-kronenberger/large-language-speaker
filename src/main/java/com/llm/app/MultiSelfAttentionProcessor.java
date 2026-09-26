package com.llm.app;

import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class MultiSelfAttentionProcessor {

  private int tokenCount;
  private int dimensions;

  public MultiSelfAttentionProcessor(int tokenCount, int dimensions) {
    this.tokenCount = tokenCount;
    this.dimensions = dimensions;
  }

  public List<List<Double>> forwardMultiHead(int multiHeadLayers, List<List<Double>> posEmbeddings) {
    System.out.println("---------SELF ATTENTION LAYER---------");

    List<List<Double>> queryVector = posEmbeddings;
    List<List<Double>> keyVector = posEmbeddings;
    List<List<Double>> valueVector = posEmbeddings;

    for (int i = 0; i < multiHeadLayers; i++) {
      // r = query^t * key
      System.out.println("Building Multi Head Attention Layer: " + i);
      RealMatrix queryMatrix = buildQueryMatrix(dimensions, dimensions, i);
      Utils.printMatrix((Array2DRowRealMatrix) queryMatrix);
      RealMatrix keyMatrix = buildKeyMatrix(dimensions, dimensions, i);
      Utils.printMatrix((Array2DRowRealMatrix) keyMatrix);

      // s = r * value
    }

    List<List<Double>> attendedEmbeddings = new ArrayList<>();
    return attendedEmbeddings;
  }

  public RealMatrix buildQueryMatrix(int rows, int columns, int multiHeadLayer) {
    String queryFilePath = "./result/attention_weights/query_" + String.valueOf(multiHeadLayer);
    return Utils.buildMatrixIfNotExists(rows, columns, queryFilePath);
  }

  public RealMatrix buildKeyMatrix(int rows, int columns, int multiHeadLayer) {
    String queryFilePath = "./result/attention_weights/key_" + String.valueOf(multiHeadLayer);
    return Utils.buildMatrixIfNotExists(rows, columns, queryFilePath);
  }
}
