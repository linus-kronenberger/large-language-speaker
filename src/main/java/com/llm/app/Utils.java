package com.llm.app;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Utils {
  public static RealMatrix buildMatrixIfNotExists(int rows, int columns, String filePath) {
    // fallback: take matrix from ./model path if available
    RealMatrix fileMatrix = loadEmbeddingMatrixFromFilePath(rows, columns, filePath);
    if(fileMatrix != null) {
      return fileMatrix;
    }

    // loop for building embedding matrix from scratch
    System.out.println("Building Embedding Matrix..");
    Array2DRowRealMatrix embeddingMatrix = new Array2DRowRealMatrix(rows, columns);
    boolean matrixFinished = false;
    while (matrixFinished == false) {
      for(int i=0; i<rows; i++) {
        // embeddingMatrix.setEntry(i, 0, i);
        for (int j = 0; j < columns; j++) {
          Random r = new Random();
          double n = r.nextGaussian();
          // Printing the random Number
          System.out.println("The next Gaussian value generated is: " + n);
          embeddingMatrix.setEntry(i, j, n);
        }
      }
      matrixFinished = true;
    }

    // store matrix in file under EMBEDDING_FILE_PATH

    try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath),
        StandardOpenOption.CREATE_NEW.ordinal())) {
      for (int i = 0; i < rows ; i++) {
        for (int j = 0; j < columns ; j++) {
          writer.write(String.valueOf(embeddingMatrix.getEntry(i, j) + " "));
        }
        if(i <= rows - 2) {
          writer.newLine();
        }
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    printMatrix(embeddingMatrix);
    return embeddingMatrix;
  }

  public static RealMatrix loadEmbeddingMatrixFromFilePath(int tokenCount, int EMBEDDING_DIMENSIONS, String filePath) {
    RealMatrix loadedEmbeddingMatrix = new Array2DRowRealMatrix(tokenCount, EMBEDDING_DIMENSIONS);
    try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
      List<String> allLines = reader.readAllLines();
      if(allLines.size() < tokenCount) {
        return null;
      }
      for (int i = 0; i < allLines.size(); i++) {
        String[] embeddingRow = allLines.get(i).split(" ");
        if(embeddingRow.length != EMBEDDING_DIMENSIONS) {
          return null;
        }
        RealVector vector = new ArrayRealVector(EMBEDDING_DIMENSIONS);
        for (int j = 0; j < embeddingRow.length; j++) {
          vector.setEntry(j, Double.valueOf(embeddingRow[j]));
        }
        loadedEmbeddingMatrix.setRowVector(i, vector);
      }
    } catch (FileNotFoundException e) {
      System.out.println(e.getMessage());
      return null;
    } catch (IOException e) {
      System.out.println(e.getMessage());
      return null;
    }
    printMatrix((Array2DRowRealMatrix) loadedEmbeddingMatrix);
    return loadedEmbeddingMatrix;
  }

  public static void printMatrix(Array2DRowRealMatrix matrix) {
    System.out.println("Embedding rows: " + matrix.getRowDimension());
    System.out.println("Embedding columns: " + matrix.getColumnDimension());
    for (int i = 0; i < matrix.getRowDimension(); i++) {
      System.out.println(Arrays.toString(matrix.getRow(i)));
    }
  }
}
