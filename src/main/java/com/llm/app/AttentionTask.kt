package com.llm.app

import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.RealMatrix
import java.util.concurrent.Callable
import kotlin.math.sqrt

class AttentionTask(
    private var seed: Int,
    private val sigma: Double,
    private val head: Int,
    private val dimensions: Int,
    private val multiHeadLayers: Int,
    private val posEmbeddings: RealMatrix,
    transformerBlock: Int
) : Callable<RealMatrix?> {
    private var transformerBlock = 0
    private var queryWeights: RealMatrix? = null
    private var keyWeights: RealMatrix? = null
    private var valueWeights: RealMatrix? = null

    init {
        this.transformerBlock = transformerBlock
    }

    @Throws(Exception::class)
    override fun call(): RealMatrix {
        println("Building Multi Head Attention Layer: " + head)
        this.queryWeights = buildQueryMatrix(dimensions, (dimensions / multiHeadLayers), head)
        Utils.printMatrix(queryWeights as Array2DRowRealMatrix)
        this.keyWeights = buildKeyMatrix(dimensions, (dimensions / multiHeadLayers), head)
        Utils.printMatrix(keyWeights as Array2DRowRealMatrix)
        this.valueWeights = buildValueMatrix(dimensions, (dimensions / multiHeadLayers), head)
        Utils.printMatrix(valueWeights as Array2DRowRealMatrix)

        val queryMatrix = Utils.multipleMatrices(posEmbeddings, queryWeights!!)
        val keyMatrix = Utils.multipleMatrices(posEmbeddings, keyWeights!!)
        val valueMatrix = Utils.multipleMatrices(posEmbeddings, valueWeights!!)

        // causal attention
        val queryKeyDotProduct = Utils.multipleMatrices(queryMatrix, keyMatrix.transpose()).scalarMultiply(
            1 / sqrt(
                dimensions.toDouble() / multiHeadLayers
            )
        )
        println("query key dot product:")
        Utils.printMatrix(queryKeyDotProduct as Array2DRowRealMatrix)
        val maskedAttentionMatrix = Utils.maskMatrix(queryKeyDotProduct)
        println("masked matrix result:")
        Utils.printMatrix(maskedAttentionMatrix as Array2DRowRealMatrix)
        val softmaxAttentionResult = Utils.calculateSoftmax(maskedAttentionMatrix)
        val valueDotProductResult = Utils.multipleMatrices(softmaxAttentionResult, valueMatrix)
        println("value dot product result:")
        Utils.printMatrix(valueDotProductResult as Array2DRowRealMatrix)
        return valueDotProductResult
    }

    fun buildQueryMatrix(rows: Int, columns: Int, multiHeadLayer: Int): RealMatrix {
        val queryFilePath =
            "./model/transformer" + transformerBlock + "/attention_weights/query_" + multiHeadLayer.toString() + ".txt"
        return Utils.buildMatrixIfNotExists(seed, sigma, rows, columns, queryFilePath)
    }

    fun buildKeyMatrix(rows: Int, columns: Int, multiHeadLayer: Int): RealMatrix {
        val keyFilePath =
            "./model/transformer" + transformerBlock + "/attention_weights/key_" + multiHeadLayer.toString() + ".txt"
        this.seed += 1
        return Utils.buildMatrixIfNotExists(seed, sigma, rows, columns, keyFilePath)
    }

    fun buildValueMatrix(rows: Int, columns: Int, multiHeadLayer: Int): RealMatrix {
        val valueFilePath =
            "./model/transformer" + transformerBlock + "/attention_weights/value_" + multiHeadLayer.toString() + ".txt"
        this.seed += 1
        return Utils.buildMatrixIfNotExists(seed, sigma, rows, columns, valueFilePath)
    }
}
