package com.llm.app

import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.RealMatrix
import java.io.IOException
import java.util.*

class SoftmaxTokenPredictor(
    private val vocabulary: MutableMap<String?, Int?>,
    @kotlin.jvm.JvmField var seed: Int,
    private val sigma: Double
) {
    private var linearHead: RealMatrix? = null
    private var meanEmbeddings: RealMatrix? = null

    @kotlin.Throws(IOException::class)
    fun generateSoftmaxResult(meanEmbeddings: RealMatrix, temperature: Double): RealMatrix {
        this.meanEmbeddings = meanEmbeddings
        val predictionLinearHeadFilePath = "./model/transformer/prediction_weights/linear_head.txt"
        this.linearHead = Utils.buildMatrixIfNotExists(
            seed,
            sigma,
            meanEmbeddings.getColumnDimension(),
            vocabulary.size,
            predictionLinearHeadFilePath
        )
        this.seed += 1

        val headResult = Utils.multipleMatrices(meanEmbeddings, linearHead)
        println("calculated head result: ")
        Utils.printMatrix(headResult as Array2DRowRealMatrix, true)
        val softmaxResult = Utils.calculateSoftmax(headResult, temperature)
        println("calculated softmax result: ")
        Utils.printMatrix(softmaxResult as Array2DRowRealMatrix, true)
        return softmaxResult
    }

    fun predictNextToken(softmaxResult: RealMatrix): String? {
        var nextToken: String? = ""

        val consideredOutputs: MutableList<Double?> = ArrayList<Double?>()
        for (column in 0..<softmaxResult.getColumnDimension()) {
            var softmaxEntry = softmaxResult.getEntry(softmaxResult.getRowDimension() - 1, column)
            softmaxEntry += column.toDouble()
            consideredOutputs.add(softmaxEntry)
        }

        val random = Random()
        val predictionRandomValue = random.nextDouble()
        var cum = 0.0
        var nextTokenIndex = 0
        for (i in consideredOutputs.indices) {
            var entry: Double = consideredOutputs.get(i)!!
            if (entry - i <= 0.0) {
                continue
            }
            var index = 0
            for (j in 0..<softmaxResult.getColumnDimension()) {
                if ((entry - j - 1) <= 0) {
                    index = j
                    entry = entry - j
                    break
                }
            }
            cum += entry
            if (predictionRandomValue < cum) {
                nextTokenIndex = index
                break
            }
        }
        println("Predicted next token index: " + nextTokenIndex)
        for (i in vocabulary.keys.indices) {
            if (vocabulary.get(vocabulary.keys.toTypedArray()[i]) == nextTokenIndex) {
                nextToken = vocabulary.keys.toTypedArray()[i].toString()
                break
            }
        }
        return nextToken
    }

    @kotlin.Throws(IOException::class)
    fun backward(
        softmaxResult: RealMatrix,
        oneHotVector: RealMatrix?,
        learningRate: Double,
        trainTemperature: Double
    ): RealMatrix {
        require(!(trainTemperature <= 0)) { "temperature must be > 0 during tarining" }

        // calculate gradient
        val X = this.meanEmbeddings!!.getRowMatrix(this.meanEmbeddings!!.getRowDimension() - 1)
        val Z = softmaxResult

        val dZ = Z.subtract(oneHotVector).scalarMultiply(1.0 / trainTemperature)
        val dW = Utils.multipleMatrices(X.transpose(), dZ)
        val dX = Utils.multipleMatrices(dZ, this.linearHead!!.transpose())

        // update weight matrix
        this.linearHead = this.linearHead!!.subtract(dW.scalarMultiply(learningRate))
        Utils.overrideFileMatrix("./model/transformer/prediction_weights/linear_head.txt", this.linearHead!!)
        return dX
    }
}
