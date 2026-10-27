package com.llm.app

import org.apache.commons.math3.linear.RealMatrix
import java.io.IOException

class App {
    var prompt: String = "Niemand hat die Absicht eine Mauer zu bauen! Nur einen Raumtrenner - mit Tür!"
    val splitRegex: String = " "

    val isTRAIN: Boolean = false
    val isPARALLELIZED: Boolean = true

    val tokenCount: Int = 50
    val embeddingDimensions: Int = 10
    val attentionHeads: Int = 2
    var seedIndex: Int = 1
    val feedForwardDimensions: Int = 4 * this.embeddingDimensions
    val maxOutputTokenAmount: Int = 10
    val transformerBlockAmount: Int = 1
    val contextWindow: Int = 100
    val batchSize: Int = 32
    val trainLines: Int = 5
    val numEpochs: Int = 10

    val sigma: Double = 0.02
    val trainTemperature: Double = 1.0

    val inferenceTemperature: Double = 1.0
    val learningRate: Double = 1e-5
    val warmUpRate: Double = 0.1

    var tokenizer: PositionalTokenizer
    @JvmField
    var multiHeads: MutableList<MultiHeadAttentionProcessor?> = ArrayList<MultiHeadAttentionProcessor?>()
    @JvmField
    var feedForwards: MutableList<FeedForwardProcessor?> = ArrayList<FeedForwardProcessor?>()
    private var tokenPredictor: SoftmaxTokenPredictor? = null

    var backpropBlockIndex: Int = -1
        private set

    init {
        this.tokenizer = PositionalTokenizer(
            this.seedIndex,
            this.sigma,
            this.tokenCount,
            this.embeddingDimensions,
            this.splitRegex
        )
    }

    @Throws(IOException::class)
    fun trainModel(): Double {
        var loss = -1.0
        val trainer = Trainer(
            this.batchSize,
            this.learningRate,
            this.warmUpRate,
            this.trainLines,
            this.splitRegex,
            this.trainTemperature
        )
        for (i in 0..<this.numEpochs) {
            loss = trainer.train()
        }
        return loss
    }

    @Throws(IOException::class)
    fun generateText(prompt: String): String {
        var prompt = prompt
        var generatedText = ""
        var nextWord = ""
        var predictions = 0
        while (nextWord != "<eos>" && predictions < this.maxOutputTokenAmount) {
            if (prompt.length > this.contextWindow) {
                prompt = prompt.substring(prompt.length - this.contextWindow - 1, prompt.length - 1)
            }
            nextWord = inferenceForwardPass(prompt)
            if (nextWord == "<eos>") {
                break
            } else if (nextWord == "<null>") {
                nextWord = " "
            }
            generatedText += nextWord + " "
            prompt += nextWord + " "
            predictions++
        }
        return generatedText
    }

    @Throws(IOException::class)
    fun inferenceForwardPass(prompt: String): String {
        val posEmbeddings = tokenizer.embeddSentence(prompt)
        this.seedIndex = tokenizer.seed
        var encodingOutputEmbeddings = posEmbeddings.copy()
        for (transformerBlock in 0..<this.transformerBlockAmount) {
            val multiHeadAttentionProcessor = MultiHeadAttentionProcessor(
                seedIndex,
                this.sigma,
                this.embeddingDimensions,
                this.isPARALLELIZED
            )
            val attentionEmbeddings = multiHeadAttentionProcessor.forwardMultiHead(
                this.attentionHeads,
                encodingOutputEmbeddings!!,
                transformerBlock
            )
            this.seedIndex = multiHeadAttentionProcessor.seed
            val feedforwarder = FeedForwardProcessor(seedIndex)
            encodingOutputEmbeddings = feedforwarder.forward(
                attentionEmbeddings!!,
                this.feedForwardDimensions, transformerBlock
            )
            this.seedIndex = feedforwarder.seed
        }

        val tokenPredictor = SoftmaxTokenPredictor(
            tokenizer.vocabulary!!, this.seedIndex,
            this.sigma
        )
        val softmaxResult = tokenPredictor.generateSoftmaxResult(encodingOutputEmbeddings!!, this.inferenceTemperature)
        this.seedIndex = tokenPredictor.seed
        val nextWord: String = tokenPredictor.predictNextToken(softmaxResult)!!
        return nextWord
    }


    @Throws(IOException::class)
    fun trainForwardPass(prompt: String): RealMatrix {
        val posEmbeddings = tokenizer.embeddSentence(prompt)
        var encodingOutputEmbeddings = posEmbeddings.copy()
        for (transformerBlock in 0..<this.transformerBlockAmount) {
            val multiHeadAttentionProcessor = MultiHeadAttentionProcessor(
                seedIndex,
                this.sigma,
                this.embeddingDimensions,
                this.isPARALLELIZED
            )
            multiHeads.add(multiHeadAttentionProcessor)
            val attentionEmbeddings = multiHeadAttentionProcessor.forwardMultiHead(
                this.attentionHeads,
                encodingOutputEmbeddings!!,
                transformerBlock
            )
            seedIndex = multiHeadAttentionProcessor.seed
            val feedforwarder = FeedForwardProcessor(seedIndex)
            feedForwards.add(feedforwarder)
            encodingOutputEmbeddings = feedforwarder.forward(
                attentionEmbeddings!!,
                this.feedForwardDimensions, transformerBlock
            )
            seedIndex = feedforwarder.seed
            this.backpropBlockIndex = transformerBlock
        }

        this.tokenPredictor = SoftmaxTokenPredictor(tokenizer.vocabulary!!, this.seedIndex, this.sigma)
        val softmaxResult = tokenPredictor!!.generateSoftmaxResult(encodingOutputEmbeddings!!, this.inferenceTemperature)
        this.seedIndex = tokenPredictor!!.seed
        return softmaxResult
    }

    fun getIndexForToken(expectedToken: String?): Int {
        return tokenizer.getIndexForToken(expectedToken)
    }

    fun getTokenPredictor(): SoftmaxTokenPredictor {
        return tokenPredictor!!
    }

    fun setTokenPredictor(tokenPredictor: SoftmaxTokenPredictor) {
        this.tokenPredictor = tokenPredictor
    }

    companion object {

}
}
    @Throws(IOException::class)
    fun main(args: Array<String>) {
        val app = App()
        if (app.isTRAIN == false) {
            val tokenOutput = app.generateText(app.prompt)
            if (app.prompt.length > app.contextWindow) {
                println("prompt was bigger than the maximum context window. Had to be cut directly..")
                println(
                    "User: " + app.prompt.substring(
                        app.prompt.length - app.contextWindow - 1,
                        app.prompt.length - 1
                    )
                )
            } else {
                println("User: " + app.prompt)
            }
            println("------------------------------")
            println("Assistant: " + tokenOutput)
        } else {
            println("Training model with parameters..")
            val finalLoss = app.trainModel()
            println("------------------------------")
            println("Final mean cross entropy loss after " + app.numEpochs + " epochs: " + finalLoss)
        }
    }
