package com.llm.app

import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.RealMatrix
import java.io.IOException
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.Future

class MultiHeadAttentionProcessor(
  var seed: Int,
  private val sigma: Double,
  private val dimensions: Int,
  private val parallelized: Boolean
) {
    private var gamma: RealMatrix? = null
    private var w0Matrix: RealMatrix? = null

    @Throws(IOException::class)
    fun forwardMultiHead(multiHeadLayers: Int, posEmbeddings: RealMatrix, transformerBlock: Int): RealMatrix? {
        val gammaPath = "./model/transformer$transformerBlock/attention_weights/gamma.txt"
        this.gamma = Utils.buildMatrixIfNotExists(seed, 0.02, posEmbeddings.columnDimension, 1, gammaPath)
        this.seed += 1
        val normalized = Utils.performRMSNormalization(posEmbeddings, gamma, 1e-5)

        val executorService = Executors.newFixedThreadPool(if (parallelized) multiHeadLayers else 1)

        println("---------SELF ATTENTION LAYER---------")

        var attentionResultMatrix: RealMatrix = Array2DRowRealMatrix(posEmbeddings.rowDimension, dimensions)
        val results: MutableList<Future<RealMatrix?>> = ArrayList<Future<RealMatrix?>>()
        val multiStartTime = System.nanoTime()
        println("Measuring multi head attention time. Starting at: $multiStartTime")

        try {
            for (i in 0..<multiHeadLayers) {
                val task = AttentionTask(seed, sigma, i, dimensions, multiHeadLayers, normalized, transformerBlock)
                this.seed += 3
                results.add(executorService.submit<RealMatrix?>(task))
            }

            var startIndex = 0

            for (result in results) {
                try {
                    attentionResultMatrix =
                        Utils.concatenateMatricesHorizontally(attentionResultMatrix, result.get(), startIndex)
                    startIndex += result.get()!!.columnDimension
                } catch (e: InterruptedException) {
                    throw RuntimeException(e)
                } catch (e: ExecutionException) {
                    throw RuntimeException(e)
                }
            }
        } finally {
            executorService.shutdown()
        }

        val multiFinishTime = System.nanoTime()
        val multiElapsedTime = multiFinishTime - multiStartTime
        println("Measured multi head attention time: $multiElapsedTime")
        this.w0Matrix = Utils.buildMatrixIfNotExists(
            seed, sigma, dimensions, dimensions, "./model/transformer$transformerBlock/attention_weights/w0.txt"
        )
        this.seed += 1
        attentionResultMatrix = Utils.multipleMatrices(attentionResultMatrix, w0Matrix)
        val resultMatrix = posEmbeddings.add(attentionResultMatrix)
        return resultMatrix
    }

    fun backward(dx: RealMatrix?): RealMatrix {
        return Array2DRowRealMatrix()
    }
}
