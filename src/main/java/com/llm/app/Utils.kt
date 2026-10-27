package com.llm.app

import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.ArrayRealVector
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.RealVector
import java.io.*
import java.util.*
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

object Utils {
    fun overrideFileMatrix(filePath: String, matrix: RealMatrix) {
        println("overriding matrix in filePath: $filePath")
        // store matrix in file under filePath
        val file = File(filePath)
        val parent = file.parentFile
        if (parent != null) {
            parent.mkdirs()
        }
        val rows = matrix.rowDimension
        val columns = matrix.columnDimension
        writeMatrix(filePath, matrix, rows, columns)
    }

    fun buildMatrixIfNotExists(seed: Int, sigma: Double, rows: Int, columns: Int, filePath: String): RealMatrix {
        // fallback: take matrix from ./model path if available
        val fileMatrix = loadEmbeddingMatrixFromFilePath(rows, columns, filePath)
        if (fileMatrix != null) {
            return fileMatrix
        }

        // loop for building embedding matrix from scratch
        println("Building Embedding Matrix with SEED: $seed")
        val r = Random(seed.toLong())
        val embeddingMatrix = Array2DRowRealMatrix(rows, columns)
        var matrixFinished = false
        while (!matrixFinished) {
            for (i in 0..<rows) {
                // embeddingMatrix.setEntry(i, 0, i);
                for (j in 0..<columns) {
                    var gaussian = r.nextGaussian()
                    if (gaussian == 0.0) {
                        gaussian = Double.MIN_VALUE
                    }
                    val n = gaussian * sigma
                    embeddingMatrix.setEntry(i, j, n)
                }
            }
            matrixFinished = true
        }

        // store matrix in file under EMBEDDING_FILE_PATH
        val file = File(filePath)
        val parent = file.parentFile
        if (parent != null) {
            parent.mkdirs()
        }

        writeMatrix(filePath, embeddingMatrix, rows, columns)

        printMatrix(embeddingMatrix)
        return embeddingMatrix
    }

    fun writeMatrix(filePath: String, embeddingMatrix: RealMatrix, rows: Int, columns: Int) {
        try {
            BufferedWriter(FileWriter(filePath)).use { writer ->
                for (i in 0..<rows) {
                    for (j in 0..<columns) {
                        writer.write("${embeddingMatrix.getEntry(i, j)} ")
                    }
                    if (i <= rows - 2) {
                        writer.newLine()
                    }
                }
            }
        } catch (e: IOException) {
            throw RuntimeException(e)
        }
    }

    fun loadEmbeddingMatrixFromFilePath(tokenCount: Int, embeddingDimension: Int, filePath: String): RealMatrix? {
        val loadedEmbeddingMatrix: RealMatrix = Array2DRowRealMatrix(tokenCount, embeddingDimension)
        try {
            BufferedReader(FileReader(filePath)).use { reader ->
                val allLines = reader.readAllLines()
                if (allLines.size != tokenCount) {
                    return null
                }
                for (i in allLines.indices) {
                    val embeddingRow: Array<String?> =
                        allLines[i]!!.split(" ".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
                    if (embeddingRow.size != embeddingDimension) {
                        return null
                    }
                    val vector: RealVector = ArrayRealVector(embeddingDimension)
                    for (j in embeddingRow.indices) {
                        vector.setEntry(j, embeddingRow[j]!!.toDouble())
                    }
                    loadedEmbeddingMatrix.setRowVector(i, vector)
                }
            }
        } catch (e: FileNotFoundException) {
            println(e.message)
            return null
        } catch (e: IOException) {
            println(e.message)
            return null
        }
        printMatrix(loadedEmbeddingMatrix as Array2DRowRealMatrix)
        return loadedEmbeddingMatrix
    }

    @JvmOverloads
    fun printMatrix(matrix: Array2DRowRealMatrix, verbose: Boolean = false) {
        if (!verbose) {
            return
        }
        println("matrix rows: ${matrix.rowDimension}")
        println("matrix columns: ${matrix.columnDimension}")
        for (i in 0..<matrix.rowDimension) {
            println(matrix.getRow(i).contentToString())
        }
    }

    @Throws(IOException::class)
    fun multipleMatrices(matrix1: RealMatrix, matrix2: RealMatrix?): RealMatrix {
        // already implemented in math3 lib

        if (matrix1.columnDimension != matrix2?.rowDimension) {
            throw IOException("Two matrices can not be multiplied as matrix 1 column dimension doesn't match matrix 2 row dimension.")
        }

        val resultMatrix: RealMatrix = Array2DRowRealMatrix(matrix1.rowDimension, matrix2.columnDimension)

        for (row1 in 0..<matrix1.rowDimension) {
            for (column2 in 0..<matrix2.columnDimension) {
                var sum = 0.0
                for (column1 in 0..<matrix1.columnDimension) {
                    sum += matrix1.getEntry(row1, column1) * matrix2.getEntry(column1, column2)
                }
                resultMatrix.setEntry(row1, column2, sum)
            }
        }
        return resultMatrix
    }

    fun calculateSoftmax(matrix: RealMatrix): RealMatrix {
        println("Calculating softmax...")
        val resultMatrix: RealMatrix = Array2DRowRealMatrix(matrix.rowDimension, matrix.columnDimension)

        for (i in 0..<matrix.rowDimension) {
            var maxRowValue = matrix.getEntry(i, 0)
            for (j in 1..<matrix.columnDimension) {
                if (maxRowValue < matrix.getEntry(i, j)) {
                    maxRowValue = matrix.getEntry(i, j)
                }
            }

            var sum = 0.0
            for (j in 0..<matrix.columnDimension) {
                sum += exp(matrix.getEntry(i, j) - maxRowValue)
            }

            for (j in 0..<matrix.columnDimension) {
                val newEntry = exp(matrix.getEntry(i, j) - maxRowValue) / sum
                resultMatrix.setEntry(i, j, newEntry)
            }
        }

        printMatrix(resultMatrix as Array2DRowRealMatrix)
        return resultMatrix
    }

    fun calculateSoftmax(matrix: RealMatrix, temperature: Double): RealMatrix {
        require(!(temperature < 0)) { "temperature must be >= 0" }

        val rows = matrix.rowDimension
        val cols = matrix.columnDimension
        val resultMatrix: RealMatrix = Array2DRowRealMatrix(rows, cols)

        for (i in 0..<rows) {
            var best = 0
            for (j in 1..<cols) {
                if (matrix.getEntry(i, j) > matrix.getEntry(i, best)) {
                    best = j
                }
            }

            if (temperature == 0.0) {
                resultMatrix.setEntry(i, best, 1.0)
                continue
            }

            val maxRowValue = matrix.getEntry(i, best)
            val exps = DoubleArray(cols)
            var sum = 0.0
            for (j in 0..<cols) {
                exps[j] = exp((matrix.getEntry(i, j) - maxRowValue) / temperature)
                sum += exps[j]
            }
            for (j in 0..<cols) {
                resultMatrix.setEntry(i, j, exps[j] / sum)
            }
        }
        return resultMatrix
    }

    fun maskMatrix(matrix: RealMatrix): RealMatrix {
        println("Masking matrix.")
        for (row in 0..<matrix.rowDimension) {
            for (column in 0..<matrix.columnDimension) {
                if (row < column) {
                    matrix.setEntry(row, column, Double.NEGATIVE_INFINITY)
                }
            }
        }
        printMatrix(matrix as Array2DRowRealMatrix)
        return matrix
    }

    fun concatenateMatricesHorizontally(matrix1: RealMatrix, matrix2: RealMatrix?, startIndex: Int): RealMatrix {
        val resMatrix = matrix1.copy()
        println("Concatenating matrices.")
        printMatrix((resMatrix as Array2DRowRealMatrix?)!!)

        for (row in 0..<resMatrix.rowDimension) {
            var secondColumn = 0
            if (matrix2 != null) {
                for (column in startIndex..<startIndex + matrix2.columnDimension) {
                    resMatrix.setEntry(row, column, matrix2.getEntry(row, secondColumn))
                    secondColumn++
                }
            }
        }

        println("Calculated concatenated matrix:")
        printMatrix(resMatrix)
        return resMatrix
    }

    fun performReLuActivation(feededMatrix: RealMatrix): RealMatrix {
        val resMatrix: RealMatrix =
            Array2DRowRealMatrix(feededMatrix.rowDimension, feededMatrix.columnDimension)
        for (row in 0..<feededMatrix.rowDimension) {
            for (column in 0..<feededMatrix.columnDimension) {
                if (feededMatrix.getEntry(row, column) > 0.0) {
                    resMatrix.setEntry(row, column, feededMatrix.getEntry(row, column))
                }
            }
        }
        return resMatrix
    }

    fun performRMSNormalization(matrix: RealMatrix, gamma: RealMatrix?, epsilon: Double): RealMatrix {
        // RMS normalization (used in LLaMa/GPT models)
        val rows = matrix.rowDimension
        val cols = matrix.columnDimension
        val normalizedMatrix: RealMatrix = Array2DRowRealMatrix(rows, cols)
        for (row in 0..<rows) {
            var sumSquares = 0.0
            for (column in 0..<cols) {
                val v = matrix.getEntry(row, column)
                sumSquares += v * v
            }
            val rms = sqrt(sumSquares / cols + epsilon)
            for (column in 0..<cols) {
                if (gamma != null) {
                    normalizedMatrix.setEntry(row, column, gamma.getEntry(column, 0) * matrix.getEntry(row, column) / rms)
                }
            }
        }
        return normalizedMatrix
    }

    fun addVectors(vec1: DoubleArray, vec2: DoubleArray): DoubleArray {
        require(vec1.size == vec2.size) { "vectors of different lengths can not be added" }
        val result = DoubleArray(vec1.size)
        for (i in vec1.indices) {
            result[i] = vec1[i] + vec2[i]
        }
        return result
    }

    fun calculateCrossEntropyLoss(oneHotVector: RealMatrix, softmaxResult: RealMatrix): Double {
        var sum = 0.0
        for (i in 0..<oneHotVector.columnDimension) {
            sum += oneHotVector.getEntry(0, i) * ln(softmaxResult.getEntry(0, i))
        }
        return (-1.0) * sum
    }
}
