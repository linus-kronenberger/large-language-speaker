package com.llm.app;

import java.io.IOException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class App 
{
    final int TOKEN_COUNT = 1000;
    final int EMBEDDING_DIMENSIONS = 120;
    final int ATTENTION_HEADS = 12;
    final int START_SEED = 1;
    final double SIGMA = 0.02;
    final boolean PARALLELIZED = true;
    final int FEED_FORWARD_DIMENSIONS = 4 * EMBEDDING_DIMENSIONS;
    final int MAX_OUTPUT_TOKEN_AMOUNT = 100;
    final double TEMPERATURE = 0.3;
    final int TRANSFORMER_BLOCK_AMOUNT = 3;

    static void main() throws IOException {
        App app = new App();
        String prompt = "Niemand hat die Absicht eine Mauer zu errichten. ";
        // Double
        String result = app.generateText(prompt);
        System.out.println("User: " + prompt);
        System.out.println("------------------------------");
        System.out.println("Assistant: " + result);
    }

    public String generateText(String prompt) throws IOException {
        PositionalTokenizer tokenizer = new PositionalTokenizer(START_SEED, SIGMA, TOKEN_COUNT, EMBEDDING_DIMENSIONS);
        String generatedText = "";
        String nextWord = "";
        int predictions = 0;
        while (nextWord != "<eos>" && predictions < MAX_OUTPUT_TOKEN_AMOUNT){
            RealMatrix posEmbeddings = tokenizer.embeddSentence(prompt);
            MultiHeadAttentionProcessor
                multiHeadAttentionProcessor = new MultiHeadAttentionProcessor(START_SEED, SIGMA, EMBEDDING_DIMENSIONS, PARALLELIZED);
            FeedForwardProcessor feedforwarder = new FeedForwardProcessor(START_SEED, ATTENTION_HEADS);
            RealMatrix encodingOutputEmbeddings = posEmbeddings.copy();
            for (int transformerBlock = 0; transformerBlock < TRANSFORMER_BLOCK_AMOUNT; transformerBlock++) {
                RealMatrix attentionEmbeddings = multiHeadAttentionProcessor.forwardMultiHead(ATTENTION_HEADS, encodingOutputEmbeddings, transformerBlock);
                encodingOutputEmbeddings = feedforwarder.forward(attentionEmbeddings, FEED_FORWARD_DIMENSIONS, transformerBlock);
            }

            SoftmaxTokenPredictor tokenPredictor = new SoftmaxTokenPredictor(tokenizer.getVocabulary(), START_SEED, SIGMA, ATTENTION_HEADS);
            nextWord = tokenPredictor.predictNextToken(encodingOutputEmbeddings, TEMPERATURE);
            generatedText += nextWord + " ";
            prompt += nextWord + " ";
            predictions ++;
        }

        return generatedText;
    }
}
