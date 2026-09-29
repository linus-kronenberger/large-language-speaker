package com.llm.app;

import java.io.IOException;
import org.apache.commons.math3.linear.RealMatrix;

public class App 
{
    final int TOKEN_COUNT = 100;
    final int EMBEDDING_DIMENSIONS = 120;
    final int ATTENTION_HEADS = 12;
    final int START_SEED = 1;
    final double SIGMA = 0.02;
    final boolean PARALLELIZED = true;
    final int FEED_FORWARD_DIMENSIONS = 4 * EMBEDDING_DIMENSIONS;
    final int OUTPUT_TOKEN_AMOUNT = 100;
    final double TEMPERATURE = 1;

    public static void main( String[] args ) throws IOException {
        App app = new App();
        String prompt = "des Alterspräsidenten ";
        String result = app.generateText(prompt);
        System.out.println("User: " + prompt);
        System.out.println("------------------------------");
        System.out.println("Assistant: " + result);
    }

    public String generateText(String prompt) throws IOException {
        PositionalTokenizer tokenizer = new PositionalTokenizer(START_SEED, SIGMA, TOKEN_COUNT, EMBEDDING_DIMENSIONS);
        String generatedText = "";
        for (int i = 0; i < OUTPUT_TOKEN_AMOUNT; i++) {
            RealMatrix posEmbeddings = tokenizer.embeddSentence(prompt);
            MultiHeadAttentionProcessor
                multiHeadAttentionProcessor = new MultiHeadAttentionProcessor(START_SEED, SIGMA, EMBEDDING_DIMENSIONS, PARALLELIZED);
            RealMatrix attentionEmbeddings = multiHeadAttentionProcessor.forwardMultiHead(ATTENTION_HEADS, posEmbeddings);
            FeedForwardProcessor feedforwarder = new FeedForwardProcessor(START_SEED, ATTENTION_HEADS);
            RealMatrix meanEmbeddings = feedforwarder.forward(attentionEmbeddings, FEED_FORWARD_DIMENSIONS);
            SoftmaxTokenPredictor tokenPredictor = new SoftmaxTokenPredictor(tokenizer.getVocabulary(), START_SEED, SIGMA, ATTENTION_HEADS);
            String nextWord = tokenPredictor.predictNextToken(meanEmbeddings, TEMPERATURE);
            generatedText += nextWord + " ";
            prompt += nextWord + " ";
        }

        return generatedText;
    }
}
