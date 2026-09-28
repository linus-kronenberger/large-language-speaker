package com.llm.app;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class FeedForwardProcessor {
  private int seed;
  private int attentionHeads;
  public FeedForwardProcessor(int seed, int attentionHeads) {
    this.attentionHeads = attentionHeads;
    this.seed = seed + attentionHeads;
  }

  public RealMatrix forward(RealMatrix attentionEmbeddings) {
    System.out.println("---------FEED FORWARD LAYER---------");
    System.out.println("start seed: " + seed);
    RealMatrix feededMatrix = new Array2DRowRealMatrix(attentionEmbeddings.getRowDimension(), attentionEmbeddings.getColumnDimension());
    System.out.println("Result Embeddings after feed forward calculation: ");
    Utils.printMatrix((Array2DRowRealMatrix) feededMatrix);
    return feededMatrix;
  }
}
