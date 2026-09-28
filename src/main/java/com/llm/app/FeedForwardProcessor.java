package com.llm.app;

import java.io.IOException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class FeedForwardProcessor {
  private int seed;
  private int attentionHeads;
  public FeedForwardProcessor(int seed, int attentionHeads) {
    this.attentionHeads = attentionHeads;
    this.seed = seed + attentionHeads * 3 + 1;
  }

  public RealMatrix forward(RealMatrix attentionEmbeddings, int feedForwardActivationDimension) throws IOException {
    System.out.println("---------FEED FORWARD LAYER---------");
    System.out.println("start seed: " + seed);
    RealMatrix feededMatrix;
    String hiddenWeightsFilePath = "./result/feed_forward_weights/hidden_weights.txt";
    RealMatrix activationFeedForwardMatrix = Utils.buildMatrixIfNotExists(seed, 0.02, attentionEmbeddings.getColumnDimension(), feedForwardActivationDimension, hiddenWeightsFilePath);
    RealMatrix hiddenLayerResult = Utils.multipleMatrices(attentionEmbeddings, activationFeedForwardMatrix); // optional: Implement bias vector layer
    Utils.performReLuActivation(hiddenLayerResult);
    System.out.println("Result Embeddings after hidden layer calculation: ");
    Utils.printMatrix((Array2DRowRealMatrix) hiddenLayerResult, true);
    String finalFeedForwardFilePath = "./result/feed_forward_weights/final_weights.txt";
    RealMatrix feedForwardFinalWeights = Utils.buildMatrixIfNotExists(seed + 1, 0.02, hiddenLayerResult.getColumnDimension(), attentionEmbeddings.getColumnDimension(), finalFeedForwardFilePath);
    feededMatrix = Utils.multipleMatrices(hiddenLayerResult, feedForwardFinalWeights);
    System.out.println("Result Embeddings after feed forward calculation: ");
    Utils.printMatrix((Array2DRowRealMatrix) feededMatrix, true);
    return feededMatrix;
  }
}
