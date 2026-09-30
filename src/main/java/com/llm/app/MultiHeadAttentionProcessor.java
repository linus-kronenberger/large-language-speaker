package com.llm.app;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class MultiHeadAttentionProcessor {

  private int seed;
  private final int startSeed;
  private final double sigma;
  private final int dimensions;
  private final boolean parallelized;

  public MultiHeadAttentionProcessor(int seed, double sigma, int dimensions, boolean parallelized) {
    this.seed = seed;
    this.startSeed = seed;
    this.sigma = sigma;
    this.dimensions = dimensions;
    this.parallelized = parallelized;
  }

  public RealMatrix forwardMultiHead(int multiHeadLayers, RealMatrix posEmbeddings, int transformerBlock) throws IOException {
    RealMatrix normalized = Utils.performNormalization(posEmbeddings, 1e-5);

    ExecutorService executorService = Executors.newFixedThreadPool(parallelized ? multiHeadLayers : 1);

    System.out.println("---------SELF ATTENTION LAYER---------");

    RealMatrix attentionResultMatrix = new Array2DRowRealMatrix(posEmbeddings.getRowDimension(), dimensions);
    List<Future<RealMatrix>> results = new ArrayList<>();
    long multiStartTime = System.nanoTime();
    System.out.println("Measuring multi head attention time. Starting at: " + multiStartTime);

    try {
      for (int i = 0; i < multiHeadLayers; i++) {
        seed = startSeed + i * 3;
        AttentionTask task = new AttentionTask(seed, sigma, i, dimensions, multiHeadLayers, normalized, transformerBlock);
        results.add(executorService.submit(task));
      }

      for (Future<RealMatrix> result : results) {
        try {
          attentionResultMatrix = Utils.concatenateMatricesHorizontally(attentionResultMatrix, result.get());
        } catch (InterruptedException | ExecutionException e) {
          throw new RuntimeException(e);
        }
      }
    } finally {
      executorService.shutdown();
    }

    long multiFinishTime = System.nanoTime();
    long multiElapsedTime = multiFinishTime - multiStartTime;
    System.out.println("Measured multi head attention time: " + multiElapsedTime);
    seed += 1;
    attentionResultMatrix = Utils.multipleMatrices(attentionResultMatrix, Utils.buildMatrixIfNotExists(seed, sigma, dimensions, dimensions, "./model/transformer" + transformerBlock
        + "/attention_weights/w0.txt"));
    RealMatrix resultMatrix = posEmbeddings.add(attentionResultMatrix);
    return resultMatrix;
  }
}
