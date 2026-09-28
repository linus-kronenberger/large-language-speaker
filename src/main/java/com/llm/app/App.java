package com.llm.app;

import java.io.IOException;
import org.apache.commons.math3.linear.RealMatrix;

public class App 
{
    final int TOKEN_COUNT = 100;
    final int EMBEDDING_DIMENSIONS = 64;
    final int ATTENTION_HEADS = 8;
    final int seed = 1;
    final double sigma = 0.02;
    final boolean parallelized = true;

    public static void main( String[] args ) throws IOException {
        App app = new App();
        String prompt = "Niemand hat die Absicht, eine Mauer zu errichten.";
        String result = app.generateText(prompt);
        System.out.println("User: " + prompt);
        System.out.println("------------------------------");
        System.out.println("Assistant: " + result);
    }

    public String generateText(String prompt) throws IOException {
        PositionalTokenizer tokenizer = new PositionalTokenizer(seed, sigma, TOKEN_COUNT, EMBEDDING_DIMENSIONS);
        RealMatrix posEmbeddings = tokenizer.embeddSentence(prompt);
        MultiHeadAttentionProcessor
            selfAttentionProcessor = new MultiHeadAttentionProcessor(seed, sigma, EMBEDDING_DIMENSIONS, parallelized);
        RealMatrix attentionEmbeddings = selfAttentionProcessor.forwardMultiHead(ATTENTION_HEADS, posEmbeddings);
        // FeedforwardProcessor feedforwarder = new FeedforwardProcessor();
        // List<List<Float>> meanEmbeddings = feedforwarder.forward(semEmbeddings);
        // WordPredictor predictor = new SoftmaxWordPredictor();
        // String nextWord = predictor.predict();
        return "";
    }
}
