package com.llm.app;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class MultiHeadAttentionProcessor {

  private int tokenCount;
  private int dimensions;

  public MultiHeadAttentionProcessor(int tokenCount, int dimensions) {
    this.tokenCount = tokenCount;
    this.dimensions = dimensions;
  }

  public List<List<Double>> forwardMultiHead(int multiHeadLayers, RealMatrix posEmbeddings) throws IOException {
    System.out.println("---------SELF ATTENTION LAYER---------");

    // todo: Parallelize self attention

    for (int i = 0; i < multiHeadLayers; i++) {
      System.out.println("Building Multi Head Attention Layer: " + i);
      RealMatrix queryWeights = buildQueryMatrix(dimensions, dimensions, i);
      Utils.printMatrix((Array2DRowRealMatrix) queryWeights);
      RealMatrix keyWeights = buildKeyMatrix(dimensions, dimensions, i);
      Utils.printMatrix((Array2DRowRealMatrix) keyWeights);
      RealMatrix valueWeights = buildValueMatrix(dimensions, dimensions, i);
      Utils.printMatrix((Array2DRowRealMatrix) valueWeights);



      RealMatrix queryMatrix = Utils.multipleMatrices(posEmbeddings, queryWeights);
      RealMatrix keyMatrix = Utils.multipleMatrices(posEmbeddings, keyWeights);
      RealMatrix valueMatrix = Utils.multipleMatrices(posEmbeddings, valueWeights);

      // bidirectional attention
      RealMatrix queryKeyDotProduct = queryMatrix.multiply(keyMatrix.transpose()).scalarMultiply(1 / Math.sqrt(
          (double) dimensions / multiHeadLayers));
      RealMatrix valueDotProduct = queryKeyDotProduct.multiply(valueMatrix);
      RealMatrix softmaxAttentionResult = Utils.calculateSoftmax(valueDotProduct);
      // transform to causal attention

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
