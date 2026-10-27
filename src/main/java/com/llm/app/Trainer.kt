package com.llm.app

import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.RealMatrix
import java.io.BufferedReader
import java.io.FileReader
import java.io.IOException
import java.util.List

class Trainer(
    private val batchSize: Int,
    private val learningRate: Double,
    private val warmupRate: Double,
    private val trainLines: Int,
    private val splitRegex: String,
    private val trainTemperature: Double
) {
    private val givenSentenceLength = 5
    private val minWordCount = 20

    @Throws(IOException::class)
    fun train(): Double {
        var meanCrossEntropyLoss = 0.0
        val app = App()
        println("Starting Training...")
        val trainingLookup = createTrainingDataset()


        // train loop
        for (i in trainingLookup.keys.indices) {
            val prompt = trainingLookup.keys.stream().toList().get(i)
            val expectedToken = trainingLookup.get(prompt)!!
                .substring(prompt.length)
                .trim { it <= ' ' }
            println("prompt: " + prompt + " expected token: " + expectedToken)
            var softmaxResult = app.trainForwardPass(prompt)
            softmaxResult = softmaxResult.getRowMatrix(softmaxResult.getRowDimension() - 1)

            // create one-hot-vector
            val correctIndex = app.getIndexForToken(expectedToken)
            val oneHotVector = createOneHotVector(correctIndex, softmaxResult.getColumnDimension())
            println("One-hot vector: ")
            Utils.printMatrix(oneHotVector as Array2DRowRealMatrix)
            var dx = app.getTokenPredictor().backward(softmaxResult, oneHotVector, learningRate, trainTemperature)
            val backwardIndex = app.backpropBlockIndex
            for (j in backwardIndex downTo 0) {
                val feedForwardProcessor = app.feedForwards.get(j)
                val dx2 = feedForwardProcessor!!.backward(dx)
                val multiHeadAttentionProcessor = app.multiHeads.get(j)
                dx = multiHeadAttentionProcessor!!.backward(dx2)
            }

            // calculate loss after training
            softmaxResult = app.trainForwardPass(prompt)
            softmaxResult = softmaxResult.getRowMatrix(softmaxResult.getRowDimension() - 1)
            val crossEntropyLoss = Utils.calculateCrossEntropyLoss(oneHotVector, softmaxResult)
            meanCrossEntropyLoss += crossEntropyLoss
            println("Mean Cross Entropy Loss (step " + i + "): " + meanCrossEntropyLoss / (i + 1))
        }

        return meanCrossEntropyLoss / trainingLookup.keys.size
    }

    fun createOneHotVector(correctIndex: Int, tokenCount: Int): RealMatrix {
        val oneHotVector: RealMatrix = Array2DRowRealMatrix(1, tokenCount)
        oneHotVector.setEntry(0, correctIndex, 1.0)
        return oneHotVector
    }

    // todo: Implement batch size
    fun createTrainingDataset(): MutableMap<String, String?> {
        val trainingDataLookup: MutableMap<String, String?> = LinkedHashMap<String, String?>()
        val DATA_FILE_PATH = "./dataset/speeches.csv"
        var line: String? = ""
        var countLines = 0
        try {
            BufferedReader(FileReader(DATA_FILE_PATH)).use { reader ->
                reader.readLine()
                reader.readLine()
                while ((reader.readLine().also { line = it }) != null) {
                    val trainingDataLookupPointer: MutableMap<String?, String?> = LinkedHashMap<String?, String?>()
                    val vocabWords = listOf(*line!!.split(splitRegex.toRegex()).dropLastWhile { it.isEmpty() }
                        .toTypedArray())

                    if (vocabWords.size < minWordCount) {
                        continue
                    }

                    // First add the given sentence
                    if (vocabWords.size == 1) {
                        continue
                    } else if (vocabWords.size > givenSentenceLength) {
                        var concatSentence = ""
                        for (i in 0..<givenSentenceLength) {
                            if (i == givenSentenceLength - 1) {
                                concatSentence += vocabWords.get(i)
                            } else {
                                concatSentence += vocabWords.get(i) + " "
                            }
                        }

                        val firstKey = concatSentence
                        val firstValue = concatSentence + " " + vocabWords.get(givenSentenceLength)

                        trainingDataLookupPointer.put(firstKey, firstValue)
                    } else {
                        continue
                    }

                    var currentKey = trainingDataLookupPointer.values.iterator().next()
                    val contextSize = if (vocabWords.size > givenSentenceLength) givenSentenceLength else 1
                    for (i in contextSize + 1..<vocabWords.size) {
                        val value = currentKey + " " + vocabWords.get(i)
                        trainingDataLookupPointer.put(currentKey, value)
                        currentKey = value
                    }
                    countLines++
                    trainingDataLookup.putAll(trainingDataLookupPointer as Map<out String, String?>)
                    if (countLines == trainLines) {
                        break
                    }
                }
            }
        } catch (e: IOException) {
            throw RuntimeException(e)
        }
        println(trainingDataLookup)
        return trainingDataLookup
    }
}
