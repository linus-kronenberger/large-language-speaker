package com.llm.app;

import java.io.IOException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class FeedForwardProcessor {
  private int seed;
  private int attentionHeads;
  private int transformerBlock = 0;

  public FeedForwardProcessor(int seed, int attentionHeads) {
    this.attentionHeads = attentionHeads;
    this.seed = seed + attentionHeads * 3 + 1;
  }

  public RealMatrix forward(RealMatrix attentionEmbeddings, int feedForwardActivationDimension, int transformerBlock) throws IOException {
    this.transformerBlock = transformerBlock;
    RealMatrix normalized = Utils.performNormalization(attentionEmbeddings, 1e-5);

    System.out.println("---------FEED FORWARD LAYER---------");
    System.out.println("start seed: " + seed);
    RealMatrix feededMatrix;
    String hiddenWeightsFilePath = "./model/transformer"+transformerBlock+"/feed_forward_weights/hidden_weights.txt";

    RealMatrix hiddenWeights = Utils.buildMatrixIfNotExists(seed, 0.02,
        normalized.getColumnDimension(), feedForwardActivationDimension, hiddenWeightsFilePath);
    RealMatrix hiddenLayerResult = Utils.multipleMatrices(normalized, hiddenWeights);
    hiddenLayerResult = Utils.performReLuActivation(hiddenLayerResult);

    System.out.println("Result Embeddings after hidden layer calculation: ");
    Utils.printMatrix((Array2DRowRealMatrix) hiddenLayerResult, true);
    String finalFeedForwardFilePath = "./model/transformer"+transformerBlock+"/feed_forward_weights/final_weights.txt";

    RealMatrix finalWeights = Utils.buildMatrixIfNotExists(seed + 1, 0.02, hiddenLayerResult.getColumnDimension(), normalized.getColumnDimension(), finalFeedForwardFilePath);
    feededMatrix = Utils.multipleMatrices(hiddenLayerResult, finalWeights);

    return attentionEmbeddings.add(feededMatrix);
  }
}
