package com.llm.app;

import java.io.IOException;
import org.apache.commons.math3.linear.RealMatrix;

public class App 
{
    final int TOKEN_COUNT = 100;
    final int EMBEDDING_DIMENSIONS = 120;
    final int ATTENTION_HEADS = 12;
    private int START_SEED = 1;
    final double SIGMA = 0.02;
    final boolean PARALLELIZED = true;
    final int FEED_FORWARD_DIMENSIONS = 4 * EMBEDDING_DIMENSIONS;
    final int MAX_OUTPUT_TOKEN_AMOUNT = 10;
    final double TEMPERATURE = 1.0;
    final int TRANSFORMER_BLOCK_AMOUNT = 1;
    final int CONTEXT_WINDOW = 100;
    final boolean TRAIN = false;
    String prompt = "Niemand hat die Absicht eine Mauer zu errichten. ";

    static void main() throws IOException {
        App app = new App();
        if(app.TRAIN == false) {
            String tokenOutput = app.generateText(app.prompt);
            if(app.prompt.length() > app.CONTEXT_WINDOW) {
                System.out.println("prompt was bigger than the maximum context window. Had to be cut directly..");
                System.out.println("User: " + app.prompt.substring(app.prompt.length() - app.CONTEXT_WINDOW - 1, app.prompt.length() - 1));
            } else {
                System.out.println("User: " + app.prompt);
            }
            System.out.println("------------------------------");
            System.out.println("Assistant: " + tokenOutput);
        } else {
         ;
        }
    }

    public String generateText(String prompt) throws IOException {
        String generatedText = "";
        String nextWord = "";
        int predictions = 0;
        while (!nextWord.equals("<eos>") && predictions < MAX_OUTPUT_TOKEN_AMOUNT){
            if(prompt.length() > CONTEXT_WINDOW) {
                prompt = prompt.substring(prompt.length() - CONTEXT_WINDOW - 1, prompt.length() - 1);
            }
            nextWord = forwardPass(prompt);
            if(nextWord.equals("<eos>")) {
                break;
            }
            generatedText += nextWord + " ";
            prompt += nextWord + " ";
            predictions ++;
        }
        return generatedText;
    }

    public String forwardPass(String prompt) throws IOException {
        PositionalTokenizer tokenizer = new PositionalTokenizer(this.START_SEED, SIGMA, TOKEN_COUNT, EMBEDDING_DIMENSIONS);
        RealMatrix posEmbeddings = tokenizer.embeddSentence(prompt);
        RealMatrix encodingOutputEmbeddings = posEmbeddings.copy();
        for (int transformerBlock = 0; transformerBlock < TRANSFORMER_BLOCK_AMOUNT; transformerBlock++) {
            int blockSeed = this.START_SEED + transformerBlock * 100;
            MultiHeadAttentionProcessor multiHeadAttentionProcessor = new MultiHeadAttentionProcessor(blockSeed, SIGMA, EMBEDDING_DIMENSIONS, PARALLELIZED);
            FeedForwardProcessor feedforwarder = new FeedForwardProcessor(blockSeed, ATTENTION_HEADS);
            RealMatrix attentionEmbeddings = multiHeadAttentionProcessor.forwardMultiHead(ATTENTION_HEADS, encodingOutputEmbeddings, transformerBlock);
            encodingOutputEmbeddings = feedforwarder.forward(attentionEmbeddings, FEED_FORWARD_DIMENSIONS, transformerBlock);
        }

        SoftmaxTokenPredictor tokenPredictor = new SoftmaxTokenPredictor(tokenizer.getVocabulary(), this.START_SEED, SIGMA, ATTENTION_HEADS);
        String nextWord = tokenPredictor.predictNextToken(encodingOutputEmbeddings, TEMPERATURE);
        this.START_SEED = 41;
        return nextWord;
    }
}
