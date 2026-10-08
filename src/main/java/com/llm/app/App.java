package com.llm.app;

import java.io.IOException;
import org.apache.commons.math3.linear.RealMatrix;

public class App 
{

    String prompt = "Niemand hat die Absicht eine Mauer zu bauen! Nur einen Raumtrenner - mit Tür!";
    final String SPLIT_REGEX = " ";

    final boolean TRAIN = false;
    final boolean PARALLELIZED = true;

    final int TOKEN_COUNT = 50;
    final int EMBEDDING_DIMENSIONS = 10;
    final int ATTENTION_HEADS = 2;
    private int seedIndex = 1;
    final int FEED_FORWARD_DIMENSIONS = 4 * EMBEDDING_DIMENSIONS;
    final int MAX_OUTPUT_TOKEN_AMOUNT = 10;
    final int TRANSFORMER_BLOCK_AMOUNT = 1;
    final int CONTEXT_WINDOW = 100;
    final int BATCH_SIZE = 32;
    final int TRAIN_LINES = 5;
    final int EPOCHS = 1;

    final double SIGMA = 0.02;
    final double TRAIN_TEMPERATURE = 1.0;

    final double TEMPERATURE = 1.0;
    final double LEARNING_RATE = 1e-5;
    final double WARM_UP_RATE = 0.1;

    private PositionalTokenizer tokenizer;

    public App() {
        this.tokenizer = new PositionalTokenizer(this.seedIndex, SIGMA, TOKEN_COUNT, EMBEDDING_DIMENSIONS, SPLIT_REGEX);
    }

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
            System.out.println("Training model with parameters..");
            double finalLoss = app.trainModel();
            System.out.println("------------------------------");
            System.out.println("Final mean cross entropy loss after " + app.EPOCHS + " epochs: " + finalLoss);
        }
    }

    public double trainModel() throws IOException {
        double loss = -1.0;
        Trainer trainer = new Trainer(BATCH_SIZE, LEARNING_RATE, WARM_UP_RATE, TRAIN_LINES, SPLIT_REGEX, TRAIN_TEMPERATURE);
        for (int i = 0; i < EPOCHS; i++) {
            loss = trainer.train();
        }
        return loss;
    }

    public String generateText(String prompt) throws IOException {
        String generatedText = "";
        String nextWord = "";
        int predictions = 0;
        while (!nextWord.equals("<eos>") && predictions < MAX_OUTPUT_TOKEN_AMOUNT){
            if(prompt.length() > CONTEXT_WINDOW) {
                prompt = prompt.substring(prompt.length() - CONTEXT_WINDOW - 1, prompt.length() - 1);
            }
            nextWord = inferenceForwardPass(prompt);
            if(nextWord.equals("<eos>")) {
                break;
            } else if(nextWord.equals("<null>")) {
                nextWord = " ";
            }
            generatedText += nextWord + " ";
            prompt += nextWord + " ";
            predictions ++;
        }
        return generatedText;
    }

    public String inferenceForwardPass(String prompt) throws IOException {
        RealMatrix posEmbeddings = tokenizer.embeddSentence(prompt);
        this.seedIndex = tokenizer.getSeed();
        RealMatrix encodingOutputEmbeddings = posEmbeddings.copy();
        for (int transformerBlock = 0; transformerBlock < TRANSFORMER_BLOCK_AMOUNT; transformerBlock++) {
            MultiHeadAttentionProcessor multiHeadAttentionProcessor = new MultiHeadAttentionProcessor(seedIndex, SIGMA, EMBEDDING_DIMENSIONS, PARALLELIZED);
            RealMatrix attentionEmbeddings = multiHeadAttentionProcessor.forwardMultiHead(ATTENTION_HEADS, encodingOutputEmbeddings, transformerBlock);
            this.seedIndex = multiHeadAttentionProcessor.getSeed();
            FeedForwardProcessor feedforwarder = new FeedForwardProcessor(seedIndex, ATTENTION_HEADS);
            encodingOutputEmbeddings = feedforwarder.forward(attentionEmbeddings, FEED_FORWARD_DIMENSIONS, transformerBlock);
            this.seedIndex = feedforwarder.getSeed();
        }

        SoftmaxTokenPredictor tokenPredictor = new SoftmaxTokenPredictor(tokenizer.getVocabulary(), this.seedIndex, SIGMA);
        RealMatrix softmaxResult = tokenPredictor.generateSoftmaxResult(encodingOutputEmbeddings, TEMPERATURE);
        this.seedIndex = tokenPredictor.getSeed();
        String nextWord = tokenPredictor.predictNextToken(softmaxResult);
        return nextWord;
    }

    public RealMatrix trainForwardPass(String prompt) throws  IOException {
        RealMatrix posEmbeddings = tokenizer.embeddSentence(prompt);
        RealMatrix encodingOutputEmbeddings = posEmbeddings.copy();
        for (int transformerBlock = 0; transformerBlock < TRANSFORMER_BLOCK_AMOUNT; transformerBlock++) {
            MultiHeadAttentionProcessor multiHeadAttentionProcessor = new MultiHeadAttentionProcessor(seedIndex, SIGMA, EMBEDDING_DIMENSIONS, PARALLELIZED);
            RealMatrix attentionEmbeddings = multiHeadAttentionProcessor.forwardMultiHead(ATTENTION_HEADS, encodingOutputEmbeddings, transformerBlock);
            seedIndex = multiHeadAttentionProcessor.getSeed();
            FeedForwardProcessor feedforwarder = new FeedForwardProcessor(seedIndex, ATTENTION_HEADS);
            encodingOutputEmbeddings = feedforwarder.forward(attentionEmbeddings, FEED_FORWARD_DIMENSIONS, transformerBlock);
            seedIndex = feedforwarder.getSeed();
        }

        SoftmaxTokenPredictor tokenPredictor = new SoftmaxTokenPredictor(tokenizer.getVocabulary(), this.seedIndex, SIGMA);
        RealMatrix softmaxResult = tokenPredictor.generateSoftmaxResult(encodingOutputEmbeddings, TEMPERATURE);
        this.seedIndex = tokenPredictor.getSeed();
        return softmaxResult;
    }

    public int getIndexForToken(String expectedToken) {
        return tokenizer.getIndexForToken(expectedToken);
    }
}
