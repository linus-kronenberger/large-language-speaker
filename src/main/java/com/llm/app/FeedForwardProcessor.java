package com.llm.app;

import java.io.IOException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class FeedForwardProcessor {
  private int seed;
  private RealMatrix gamma;
  private RealMatrix hiddenWeights;
  private RealMatrix finalWeights;

  public FeedForwardProcessor(int seed, int attentionHeads) {
    this.seed = seed;
  }

  public RealMatrix forward(RealMatrix attentionEmbeddings, int feedForwardActivationDimension, int transformerBlock) throws IOException {
    String gammaPath = "./model/transformer"+transformerBlock+"/feed_forward_weights/gamma.txt";
    this.gamma = Utils.buildMatrixIfNotExists(seed, 0.02, attentionEmbeddings.getColumnDimension(), 1, gammaPath);
    this.seed += 1;
    RealMatrix normalized = Utils.performRMSNormalization(attentionEmbeddings, gamma, 1e-5);

    System.out.println("---------FEED FORWARD LAYER---------");
    System.out.println("start seed: " + seed);
    RealMatrix feededMatrix;
    String hiddenWeightsFilePath = "./model/transformer"+transformerBlock+"/feed_forward_weights/hidden_weights.txt";

    this.hiddenWeights = Utils.buildMatrixIfNotExists(seed, 0.02,
        normalized.getColumnDimension(), feedForwardActivationDimension, hiddenWeightsFilePath);
    this.seed += 1;
    RealMatrix hiddenLayerResult = Utils.multipleMatrices(normalized, hiddenWeights);
    hiddenLayerResult = Utils.performReLuActivation(hiddenLayerResult);

    System.out.println("Result Embeddings after hidden layer calculation: ");
    Utils.printMatrix((Array2DRowRealMatrix) hiddenLayerResult, true);
    String finalFeedForwardFilePath = "./model/transformer"+transformerBlock+"/feed_forward_weights/final_weights.txt";

    this.finalWeights = Utils.buildMatrixIfNotExists(seed, 0.02, hiddenLayerResult.getColumnDimension(), normalized.getColumnDimension(), finalFeedForwardFilePath);
    this.seed += 1;
    feededMatrix = Utils.multipleMatrices(hiddenLayerResult, finalWeights);
    RealMatrix resultMatrix = attentionEmbeddings.add(feededMatrix);
    return resultMatrix;
  }

  public int getSeed() {
    return seed;
  }
}
