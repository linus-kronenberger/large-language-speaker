package com.llm.app;

import java.io.IOException;
import java.util.List;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Hello world!
 *
 */
public class App 
{
    int TOKEN_COUNT = 100;
    int EMBEDDING_DIMENSIONS = 64;

    public static void main( String[] args ) throws IOException {
        App app = new App();
        String prompt = "Niemand hat die Absicht, eine Mauer zu errichten.";
        String result = app.generateText(prompt);
        System.out.println("User: " + prompt);
        System.out.println("------------------------------");
        System.out.println("Assistant: " + result);
    }

    public String generateText(String prompt) throws IOException {
        PositionalTokenizer tokenizer = new PositionalTokenizer(TOKEN_COUNT, EMBEDDING_DIMENSIONS);
        RealMatrix posEmbeddings = tokenizer.embeddSentence(prompt);
        MultiHeadAttentionProcessor
            selfAttentionProcessor = new MultiHeadAttentionProcessor(TOKEN_COUNT, EMBEDDING_DIMENSIONS);
        selfAttentionProcessor.forwardMultiHead(5, posEmbeddings);
        // FeedforwardProcessor feedforwarder = new FeedforwardProcessor();
        // List<List<Float>> meanEmbeddings = feedforwarder.forward(semEmbeddings);
        // WordPredictor predictor = new SoftmaxWordPredictor();
        // String nextWord = predictor.predict();
        return "";
    }
}
