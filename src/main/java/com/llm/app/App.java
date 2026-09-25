package com.llm.app;

import java.util.List;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        App app = new App();
        String prompt = "Niemand hat die Absicht eine Mauer zu errichten.";
        String result = app.generateText(prompt);
        System.out.println("User: " + prompt);
        System.out.println("------------------------------");
        System.out.println("Assistant: " + result);
    }

    public String generateText(String prompt) {
        PositionalTokenizer tokenizer = new PositionalTokenizer();
        List<List<Double>> posEmbeddings = tokenizer.embeddSentence(prompt);
        MultiSelfAttentionProcessor selfAttentionProcessor = new MultiSelfAttentionProcessor(posEmbeddings);
        // List<List<Float>> semEmbeddings = selfAttentionProcessor.embedd(List<List<Float>>));
        // FeedforwardProcessor feedforwarder = new FeedforwardProcessor();
        // List<List<Float>> meanEmbeddings = feedforwarder.forward(semEmbeddings);
        // WordPredictor predictor = new WordPredictor();
        // String nextWord = predictor.predict();
        return "";
    }
}
