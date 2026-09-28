package com.llm.app;

import java.util.concurrent.Callable;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class AttentionTask implements Callable<RealMatrix> {

  private int head;
  private int dimensions;
  private int multiHeadLayers;
  private RealMatrix posEmbeddings;
  private int seed;
  private double sigma;

  public AttentionTask(int seed, double sigma, int head, int dimensions, int multiHeadLayers, RealMatrix posEmbeddings) {
    this.seed = seed;
    this.sigma = sigma;
    this.head = head;
    this.dimensions = dimensions;
    this.multiHeadLayers = multiHeadLayers;
    this.posEmbeddings = posEmbeddings;
  }

  @Override
  public RealMatrix call() throws Exception {
    System.out.println("Building Multi Head Attention Layer: " + head);
    RealMatrix queryWeights = buildQueryMatrix(dimensions, (dimensions / multiHeadLayers), head);
    Utils.printMatrix((Array2DRowRealMatrix) queryWeights);
    RealMatrix keyWeights = buildKeyMatrix(dimensions, (dimensions / multiHeadLayers), head);
    Utils.printMatrix((Array2DRowRealMatrix) keyWeights);
    RealMatrix valueWeights = buildValueMatrix(dimensions, (dimensions / multiHeadLayers), head);
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
    return valueDotProductResult;
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
