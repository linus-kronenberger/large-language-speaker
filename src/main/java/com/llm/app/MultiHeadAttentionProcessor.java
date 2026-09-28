package com.llm.app;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class MultiHeadAttentionProcessor {

  private int seed;
  private double sigma;
  private int dimensions;

  public MultiHeadAttentionProcessor(int seed, double sigma, int dimensions) {
    this.seed = seed;
    this.sigma = sigma;
    this.dimensions = dimensions;
  }

  public RealMatrix forwardMultiHead(int multiHeadLayers, RealMatrix posEmbeddings) throws IOException {
    System.out.println("---------SELF ATTENTION LAYER---------");

    // todo: Parallelize self attention

    RealMatrix attentionResultMatrix = new Array2DRowRealMatrix(posEmbeddings.getRowDimension(), dimensions);

    for (int i = 0; i < multiHeadLayers; i++) {
      System.out.println("Building Multi Head Attention Layer: " + i);
      RealMatrix queryWeights = buildQueryMatrix(dimensions, (dimensions / multiHeadLayers), i);
      Utils.printMatrix((Array2DRowRealMatrix) queryWeights);
      RealMatrix keyWeights = buildKeyMatrix(dimensions, (dimensions / multiHeadLayers), i);
      Utils.printMatrix((Array2DRowRealMatrix) keyWeights);
      RealMatrix valueWeights = buildValueMatrix(dimensions, (dimensions / multiHeadLayers), i);
      Utils.printMatrix((Array2DRowRealMatrix) valueWeights);

      RealMatrix queryMatrix = Utils.multipleMatrices(posEmbeddings, queryWeights);
      RealMatrix keyMatrix = Utils.multipleMatrices(posEmbeddings, keyWeights);
      RealMatrix valueMatrix = Utils.multipleMatrices(posEmbeddings, valueWeights);

      // causal attention
      RealMatrix queryKeyDotProduct = Utils.multipleMatrices(queryMatrix, keyMatrix.transpose()).scalarMultiply(1 / Math.sqrt(
          (double) dimensions / multiHeadLayers));
      System.out.println("query key dot product:");
      Utils.printMatrix((Array2DRowRealMatrix) queryKeyDotProduct);
      RealMatrix maskedAttentionMatrix = Utils.maskMatrix(queryKeyDotProduct);
      System.out.println("masked matrix result:");
      Utils.printMatrix((Array2DRowRealMatrix) maskedAttentionMatrix);
      RealMatrix softmaxAttentionResult = Utils.calculateSoftmax(maskedAttentionMatrix);
      RealMatrix valueDotProductResult = Utils.multipleMatrices(softmaxAttentionResult, valueMatrix);
      System.out.println("value dot product result:");
      Utils.printMatrix((Array2DRowRealMatrix) valueDotProductResult);
      Utils.concatenateMatricesHorizontally(attentionResultMatrix, valueDotProductResult);
    }

    return attentionResultMatrix;
  }

  public RealMatrix buildQueryMatrix(int rows, int columns, int multiHeadLayer) {
    String queryFilePath = "./result/attention_weights/query_" + String.valueOf(multiHeadLayer);
    seed += 1;
    return Utils.buildMatrixIfNotExists(seed, sigma, rows, columns, queryFilePath);
  }

  public RealMatrix buildKeyMatrix(int rows, int columns, int multiHeadLayer) {
    String queryFilePath = "./result/attention_weights/key_" + String.valueOf(multiHeadLayer);
    seed += 1;
    return Utils.buildMatrixIfNotExists(seed, sigma, rows, columns, queryFilePath);
  }

  public RealMatrix buildValueMatrix(int rows, int columns, int multiHeadLayer) {
    String queryFilePath = "./result/attention_weights/value_" + String.valueOf(multiHeadLayer);
    seed += 1;
    return Utils.buildMatrixIfNotExists(seed, sigma, rows, columns, queryFilePath);
  }
}
