package com.llm.app

import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.RealMatrix
import java.io.BufferedReader
import java.io.FileReader
import java.io.IOException
import java.util.concurrent.atomic.AtomicInteger
import java.util.function.Consumer
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.pow
import kotlin.math.sin

class PositionalTokenizer(
    var seed: Int,
    private val sigma: Double,
    private val tokenCount: Int,
    private val dimensions: Int,
    private val splitRegex: String
) {
    var vocabulary: MutableMap<String?, Int?>? = null
        private set
    private var embeddingMatrix: RealMatrix? = null

    init {
        println("---------POSITIONAL TOKENIZER LAYER---------")
    }

    @Throws(IOException::class)
    fun embeddSentence(sentence: String): RealMatrix {
        this.vocabulary = buildVocabulary()
        val embeddingFilePath = "./model/transformer/embedding_lookup/embedding_matrix.txt"
        this.embeddingMatrix = Utils.buildMatrixIfNotExists(
            seed, sigma, vocabulary!!.keys.size, dimensions,
            embeddingFilePath
        )
        this.seed += 1
        val wordList =
            listOf<String?>(*sentence.split(" |(\\n)".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray())
        val tokenPosEmbeddings: RealMatrix = Array2DRowRealMatrix(wordList.size, dimensions)


        val positionEncoding = AtomicInteger()
        val counter = AtomicInteger(0)
        wordList.forEach(Consumer { word: String? ->

            // tokenize
            val tokenEmbedding = wordEncode(positionEncoding.get(), word, vocabulary!!, embeddingMatrix!!)
            println("word: $word")
            print("embedding: ")
            print("[")

            // generate positional embedding
            for (i in 0..<dimensions) {
                print("${tokenEmbedding[i]} ")
                tokenPosEmbeddings.setEntry(counter.get(), i, tokenEmbedding[i])
            }
            println("]")
            println()


            positionEncoding.getAndIncrement()
            counter.getAndIncrement()
        })
        return tokenPosEmbeddings
    }

    fun wordEncode(
        position: Int,
        word: String?,
        vocabulary: MutableMap<String?, Int?>,
        embeddingMatrix: RealMatrix
    ): DoubleArray {
        val wordEmbeddingVector: DoubleArray
        val positionalEmbedding = DoubleArray(dimensions)
        if (vocabulary[word] != null) {
            val tokenNumber: Int = vocabulary[word]!!
            wordEmbeddingVector = embeddingMatrix.getRow(tokenNumber)
        } else {
            val tokenNumber: Int = vocabulary["<null>"]!!
            println("Word: $word not known.")
            wordEmbeddingVector = embeddingMatrix.getRow(tokenNumber)
        }
        for (i in positionalEmbedding.indices) {
            val exponent = (2.0 * floor(i / 2.0)) / dimensions
            if (i % 2 == 0) {
                // even indices use sinusoidal pos embeddings
                positionalEmbedding[i] = sin(position / 10000.0.pow(exponent))
            } else {
                // odd indices use cosinoidal pos embeddings
                positionalEmbedding[i] = cos(position / 10000.0.pow(exponent))
            }
        }
        return Utils.addVectors(wordEmbeddingVector, positionalEmbedding)
    }

    fun buildVocabulary(): MutableMap<String?, Int?> {
        println("Building Vocabulary..")
        val vocabulary: MutableMap<String?, Int?> = LinkedHashMap()
        var line: String?
        val vocabLimit = this.tokenCount // max value is 25605402
        val count = AtomicInteger(0)
        vocabulary["<null>"] = 0
        vocabulary["<eos>"] = 1
        val wordIndex = AtomicInteger(2)
        val filePath = "./dataset/speeches.csv"
        try {
            BufferedReader(FileReader(filePath)).use { reader ->
                while ((reader.readLine().also { line = it }) != null) {
                    println(line)
                    val vocabWords = listOf<String?>(*line!!.split(splitRegex.toRegex()).dropLastWhile { it.isEmpty() }
                        .toTypedArray())
                    vocabWords.forEach(Consumer { word: String? ->
                        if (!vocabulary.containsKey(word)) {
                            vocabulary[word] = wordIndex.get()
                            println("Added token to vocabulary: $wordIndex   $word")
                            wordIndex.getAndIncrement()
                        }
                    })
                    count.getAndIncrement()
                    if (count.toInt() == vocabLimit) {
                        break
                    }
                }
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }
        println("Amount of tokens in library: ${vocabulary.keys.size}")
        return vocabulary
    }

    fun indexForToken(expectedToken: String?): Int {
        if (vocabulary == null) {
            this.vocabulary = buildVocabulary()
        }
        if (vocabulary!![expectedToken] == null) {
            return 0
        }
        return vocabulary!![expectedToken]!!
    }
}
