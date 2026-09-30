package com.llm.app;

public class Trainer {
  private int batchSize;
  private double learningRate;
  private double warmupRate;

  public Trainer(int BATCH_SIZE, double LEARNING_RATE, double WARM_UP_RATE, int TRAIN_LINES) {
    this.batchSize = BATCH_SIZE;
    this.learningRate = LEARNING_RATE;
    this.warmupRate = WARM_UP_RATE;
  }

  double train() {
    double loss = 0.0;
    return loss;
  }
}
