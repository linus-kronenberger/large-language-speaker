package com.llm.app;

import java.util.ArrayList;
import java.util.List;
import jdk.jshell.execution.Util;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class MultiHeadAttentionProcessor {

  private int tokenCount;
  private int dimensions;

  public MultiHeadAttentionProcessor(int tokenCount, int dimensions) {
    this.tokenCount = tokenCount;
    this.dimensions = dimensions;
  }

  public List<List<Double>> forwardMultiHead(int multiHeadLayers, RealMatrix posEmbeddings) {
    System.out.println("---------SELF ATTENTION LAYER---------");

    for (int i = 0; i < multiHeadLayers; i++) {
      // r = query^t * key
      System.out.println("Building Multi Head Attention Layer: " + i);
      RealMatrix queryMatrix = buildQueryMatrix(dimensions, dimensions, i);
      Utils.printMatrix((Array2DRowRealMatrix) queryMatrix);
      RealMatrix keyMatrix = buildKeyMatrix(dimensions, dimensions, i);
      Utils.printMatrix((Array2DRowRealMatrix) keyMatrix);
      RealMatrix valueMatrix = buildValueMatrix(dimensions, dimensions, i);
      Utils.printMatrix((Array2DRowRealMatrix) valueMatrix);

      // causal attention

      for (int j = 0; j < posEmbeddings.getRowDimension(); j++) {

          RealVector queryVector = new ArrayRealVector(dimensions);
          RealVector keyVector =  new ArrayRealVector(dimensions);
          RealVector valueVector =  new ArrayRealVector(dimensions);

          Utils.multiplyTransposedVectorWithMatrix(queryVector, queryMatrix);
          Utils.multiplyTransposedVectorWithMatrix(keyVector, keyMatrix);
          Utils.multiplyTransposedVectorWithMatrix(valueVector, valueMatrix);

        // queryVector.setEntry(i, posEmbeddings.get(i));
      }

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

  public RealMatrix buildValueMatrix(int rows, int columns, int multiHeadLayer) {
    String queryFilePath = "./result/attention_weights/value_" + String.valueOf(multiHeadLayer);
    return Utils.buildMatrixIfNotExists(rows, columns, queryFilePath);
  }
}
