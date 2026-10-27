package com.llm.app

import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.RealMatrix
import java.io.IOException

class FeedForwardProcessor(@JvmField var seed: Int) {
    private var gamma: RealMatrix? = null
    private var hiddenWeights: RealMatrix? = null
    private var finalWeights: RealMatrix? = null

    @Throws(IOException::class)
    fun forward(
        attentionEmbeddings: RealMatrix,
        feedForwardActivationDimension: Int,
        transformerBlock: Int
    ): RealMatrix? {
        val gammaPath = "./model/transformer" + transformerBlock + "/feed_forward_weights/gamma.txt"
        this.gamma = Utils.buildMatrixIfNotExists(seed, 0.02, attentionEmbeddings.getColumnDimension(), 1, gammaPath)
        this.seed += 1
        val normalized = Utils.performRMSNormalization(attentionEmbeddings, gamma, 1e-5)

        println("---------FEED FORWARD LAYER---------")
        println("start seed: " + seed)
        val feededMatrix: RealMatrix?
        val hiddenWeightsFilePath =
            "./model/transformer" + transformerBlock + "/feed_forward_weights/hidden_weights.txt"

        this.hiddenWeights = Utils.buildMatrixIfNotExists(
            seed, 0.02,
            normalized.getColumnDimension(), feedForwardActivationDimension, hiddenWeightsFilePath
        )
        this.seed += 1
        var hiddenLayerResult = Utils.multipleMatrices(normalized, hiddenWeights)
        hiddenLayerResult = Utils.performReLuActivation(hiddenLayerResult)

        println("Result Embeddings after hidden layer calculation: ")
        Utils.printMatrix(hiddenLayerResult as Array2DRowRealMatrix, true)
        val finalFeedForwardFilePath =
            "./model/transformer" + transformerBlock + "/feed_forward_weights/final_weights.txt"

        this.finalWeights = Utils.buildMatrixIfNotExists(
            seed,
            0.02,
            hiddenLayerResult.getColumnDimension(),
            normalized.getColumnDimension(),
            finalFeedForwardFilePath
        )
        this.seed += 1
        feededMatrix = Utils.multipleMatrices(hiddenLayerResult, finalWeights)
        val resultMatrix = attentionEmbeddings.add(feededMatrix)
        return resultMatrix
    }

    fun backward(dx: RealMatrix?): RealMatrix {
        return Array2DRowRealMatrix()
    }
}
