package com.llm.app;

import java.util.List;

public class Main {
  static void main() {

  }

  public String generateText(String prompt) {
    PositionalTokenizer tokenizer = new PositionalTokenizer();
    List<List<Float>> posEmbeddings = tokenizer.embeddSentence(prompt);
    // MultiSelfAttentionProcessor selfAttentionProcessor = new MultiSelfAttentionProcessor();
    // List<List<Float>> semEmbeddings = selfAttentionProcessor.embedd(List<List<Float>>));
    // FeedforwardProcessor feedforwarder = new FeedforwardProcessor();
    // List<List<Float>> meanEmbeddings = feedforwarder.forward(semEmbeddings);
    // WordPredictor predictor = new WordPredictor();
    // String nextWord = predictor.predict();
    return "";
  }
}
