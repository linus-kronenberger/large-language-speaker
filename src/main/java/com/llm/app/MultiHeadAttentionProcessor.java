package com.llm.app;

import java.io.IOException;
import java.sql.Time;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class MultiHeadAttentionProcessor {

  private int seed;
  private double sigma;
  private int dimensions;
  private boolean parallelized;

  public MultiHeadAttentionProcessor(int seed, double sigma, int dimensions, boolean parallelized) {
    this.seed = seed;
    this.sigma = sigma;
    this.dimensions = dimensions;
    this.parallelized = parallelized;
  }

  public RealMatrix forwardMultiHead(int multiHeadLayers, RealMatrix posEmbeddings) throws IOException {
    RealMatrix normalized = Utils.performNormalization(posEmbeddings, 1e-5);

    ExecutorService executorService = Executors.newFixedThreadPool(parallelized ? multiHeadLayers : 1);

    System.out.println("---------SELF ATTENTION LAYER---------");

    RealMatrix attentionResultMatrix = new Array2DRowRealMatrix(posEmbeddings.getRowDimension(), dimensions);
    List<Future<RealMatrix>> results = new ArrayList<>();
    long multiStartTime = System.nanoTime();
    System.out.println("Measuring multi head attention time. Starting at: " + multiStartTime);

    try {
      for (int i = 0; i < multiHeadLayers; i++) {
        AttentionTask task = new AttentionTask(seed + i * 3, sigma, i, dimensions, multiHeadLayers, normalized);
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
    attentionResultMatrix = Utils.multipleMatrices(attentionResultMatrix, Utils.buildMatrixIfNotExists(seed, sigma, dimensions, dimensions, "./result/attention_weights/wo.txt"));
    return posEmbeddings.add(attentionResultMatrix);
  }
}
