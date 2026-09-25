package com.llm.app;

import java.util.List;

public class MultiSelfAttentionProcessor {
  public MultiSelfAttentionProcessor(List<List<Double>> posEmbeddings) {
    System.out.println("---------SELF ATTENTION LAYER---------");
    posEmbeddings.forEach(embedds -> {
      embedds.forEach(embedd -> {
        System.out.print(embedd.toString());
        System.out.print(" ");
      });
      System.out.println("\n");
    });
  }
}
