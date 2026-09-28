package com.llm.app;

import java.io.IOException;
import org.apache.commons.math3.linear.RealMatrix;

public class App 
{
    final int TOKEN_COUNT = 100;
    final int EMBEDDING_DIMENSIONS = 64;
    final int ATTENTION_HEADS = 8;
    final int START_SEED = 1;
    final double SIGMA = 0.02;
    final boolean PARALLELIZED = true;

    public static void main( String[] args ) throws IOException {
        App app = new App();
        String prompt = "Niemand hat die Absicht, eine Mauer zu errichten.";
        String result = app.generateText(prompt);
        System.out.println("User: " + prompt);
        System.out.println("------------------------------");
        System.out.println("Assistant: " + result);
    }

    public String generateText(String prompt) throws IOException {
        PositionalTokenizer tokenizer = new PositionalTokenizer(START_SEED, SIGMA, TOKEN_COUNT, EMBEDDING_DIMENSIONS);
        RealMatrix posEmbeddings = tokenizer.embeddSentence(prompt);
        MultiHeadAttentionProcessor
            multiHeadAttentionProcessor = new MultiHeadAttentionProcessor(START_SEED, SIGMA, EMBEDDING_DIMENSIONS, PARALLELIZED);
        RealMatrix attentionEmbeddings = multiHeadAttentionProcessor.forwardMultiHead(ATTENTION_HEADS, posEmbeddings);
        FeedForwardProcessor feedforwarder = new FeedForwardProcessor();
        // List<List<Float>> meanEmbeddings = feedforwarder.forward(semEmbeddings);
        // WordPredictor predictor = new SoftmaxWordPredictor();
        // String nextWord = predictor.predict();
        return "";
    }
}
