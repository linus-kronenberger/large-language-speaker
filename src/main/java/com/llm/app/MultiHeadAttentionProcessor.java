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

  public RealMatrix forwardMultiHead(int multiHeadLayers, RealMatrix posEmbeddings) {
    ExecutorService executorService = Executors.newFixedThreadPool(1);
    if(parallelized == true) {
      executorService = Executors.newFixedThreadPool(multiHeadLayers);
    }

    System.out.println("---------SELF ATTENTION LAYER---------");

    RealMatrix attentionResultMatrix = new Array2DRowRealMatrix(posEmbeddings.getRowDimension(), dimensions);
    List<Future<RealMatrix>> results = new ArrayList<>();
    long multiStartTime = System.nanoTime();
    System.out.println("Measuring multi head attention time. Starting at: " + multiStartTime);
    for (int i = 0; i < multiHeadLayers; i++) {
      seed ++;
      AttentionTask task = new AttentionTask(seed, sigma, i, dimensions, multiHeadLayers, posEmbeddings);
      Future<RealMatrix> futureDotValueResult = executorService.submit(task);
      results.add(futureDotValueResult);
    }

    results.forEach(result -> {
      try {
        Utils.concatenateMatricesHorizontally(attentionResultMatrix, result.get());
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      } catch (ExecutionException e) {
        throw new RuntimeException(e);
      }
    });

    long multiFinishTime = System.nanoTime();
    long multiElapsedTime = multiFinishTime - multiStartTime;
    System.out.println("Measured multi head attention time: " + multiElapsedTime);

    return attentionResultMatrix;
  }
}
