package com.llm.app;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Utils {
  public static RealMatrix buildMatrixIfNotExists(int seed, double sigma, int rows, int columns, String filePath) {
    // fallback: take matrix from ./model path if available
    RealMatrix fileMatrix = loadEmbeddingMatrixFromFilePath(rows, columns, filePath);
    if(fileMatrix != null) {
      return fileMatrix;
    }

    // loop for building embedding matrix from scratch
    System.out.println("Building Embedding Matrix with SEED: " + seed);
    Random r = new Random(seed);
    Array2DRowRealMatrix embeddingMatrix = new Array2DRowRealMatrix(rows, columns);
    boolean matrixFinished = false;
    while (matrixFinished == false) {
      for(int i=0; i < rows; i++) {
        // embeddingMatrix.setEntry(i, 0, i);
        for (int j = 0; j < columns; j++) {
          double n = r.nextGaussian() * sigma;
          embeddingMatrix.setEntry(i, j, n);
        }
      }
      matrixFinished = true;
    }

    // store matrix in file under EMBEDDING_FILE_PATH
    File file = new File(filePath);
    File parent = file.getParentFile();
    if(parent != null) {
      parent.mkdirs();
    }
    try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
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

  public static void printMatrix(Array2DRowRealMatrix matrix, boolean verbose) {
    if(verbose == false) {
      return;
    }
    System.out.println("matrix rows: " + matrix.getRowDimension());
    System.out.println("matrix columns: " + matrix.getColumnDimension());
    for (int i = 0; i < matrix.getRowDimension(); i++) {
      System.out.println(Arrays.toString(matrix.getRow(i)));
    }
  }

  public static void printMatrix(Array2DRowRealMatrix matrix) {
    printMatrix(matrix, false);
  }

  public static RealVector multiplyTransposedVectorWithMatrix(RealVector vector, RealMatrix matrix) {
    RealVector resultVector = new ArrayRealVector(vector.getDimension());
    for (int i = 0; i < vector.getDimension(); i++) {
      double sum = 0.0;
      for (int j = 0; j < vector.getDimension(); j++) {
        sum += vector.getEntry(i) * matrix.getEntry(i, j);
      }
      resultVector.setEntry(i, sum);
    }
    return resultVector;
  }
  
  public static RealMatrix multipleMatrices(RealMatrix matrix1, RealMatrix matrix2) throws IOException {
    // already implemented in math3 lib

    if(matrix1.getColumnDimension() != matrix2.getRowDimension()) {
      throw new IOException("Two matrices can not be multiplied as matrix 1 column dimension doesn't match matrix 2 row dimension.");
    }

    RealMatrix resultMatrix = new Array2DRowRealMatrix(matrix1.getRowDimension(), matrix2.getColumnDimension());

    for (int row1 = 0; row1 < matrix1.getRowDimension(); row1++) {
      for (int column2 = 0; column2 < matrix2.getColumnDimension(); column2++) {
        double sum = 0;
        for (int column1 = 0; column1 < matrix1.getColumnDimension(); column1++) {
          sum += matrix1.getEntry(row1, column1) * matrix2.getEntry(column1, column2);
        }
        resultMatrix.setEntry(row1, column2, sum);
      }
    }
    return resultMatrix;
  }

  public static RealMatrix calculateSoftmax(RealMatrix matrix) {
    System.out.println("Calculating softmax...");
    RealMatrix resultMatrix = new Array2DRowRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());

    for (int i = 0; i < matrix.getRowDimension(); i++) {
      double maxRowValue = matrix.getEntry(i, 0);
      for (int j = 1; j < matrix.getColumnDimension(); j++) {
        if (maxRowValue < matrix.getEntry(i, j)) {
          maxRowValue = matrix.getEntry(i, j);
        }
      }

      double sum = 0.0;
      for (int j = 0; j < matrix.getColumnDimension(); j++) {
        sum += Math.exp(matrix.getEntry(i, j) - maxRowValue);
      }

      for (int j = 0; j < matrix.getColumnDimension(); j++) {
        double newEntry = Math.exp(matrix.getEntry(i, j) - maxRowValue) / sum;
        resultMatrix.setEntry(i, j, newEntry);
      }
    }

    printMatrix((Array2DRowRealMatrix) resultMatrix);
    return resultMatrix;
  }

  public static RealMatrix calculateSoftmax(RealMatrix matrix, double temperature) {
    if (temperature < 0) {
      throw new IllegalArgumentException("temperature must be >= 0");
    }

    int rows = matrix.getRowDimension();
    int cols = matrix.getColumnDimension();
    RealMatrix resultMatrix = new Array2DRowRealMatrix(rows, cols);

    for (int i = 0; i < rows; i++) {

      int best = 0;
      for (int j = 1; j < cols; j++) {
        if (matrix.getEntry(i, j) > matrix.getEntry(i, best)) {
          best = j;
        }
      }

      if (temperature == 0.0) {
        resultMatrix.setEntry(i, best, 1.0);
        continue;
      }

      double maxRowValue = matrix.getEntry(i, best);
      double[] exps = new double[cols];
      double sum = 0.0;
      for (int j = 0; j < cols; j++) {
        exps[j] = Math.exp((matrix.getEntry(i, j) - maxRowValue) / temperature);
        sum += exps[j];
      }
      for (int j = 0; j < cols; j++) {
        resultMatrix.setEntry(i, j, exps[j] / sum);
      }
    }
    return resultMatrix;
  }

  public static RealMatrix maskMatrix(RealMatrix matrix) {
    System.out.println("Masking matrix.");
    for (int row = 0; row < matrix.getRowDimension(); row++) {
      for (int column = 0; column < matrix.getColumnDimension(); column++) {
        if(row < column) {
          matrix.setEntry(row, column, Double.NEGATIVE_INFINITY);
        }
      }
    }
    printMatrix((Array2DRowRealMatrix) matrix);
    return matrix;
  }

  public static RealMatrix concatenateMatricesHorizontally(RealMatrix matrix1, RealMatrix matrix2) {
    RealMatrix resMatrix = matrix1.copy();
    System.out.println("Concatenating matrices.");
    printMatrix((Array2DRowRealMatrix) resMatrix);

    // first search for index
    int emptyIndex = -1;
    for (int i = 0; i < resMatrix.getColumnDimension(); i++) {
      if(resMatrix.getEntry(0, i) == 0.0) {
        emptyIndex = i;
        break;
      }
    }
    if(emptyIndex == -1) {
      System.out.println("Matrix 1 is already full. Concatenation with matrix 2 not possible.");
    }

    for (int row = 0; row < resMatrix.getRowDimension(); row++) {
      int secondColumn = 0;
      for (int column = emptyIndex; column < emptyIndex + matrix2.getColumnDimension(); column++) {
        resMatrix.setEntry(row, column, matrix2.getEntry(row, secondColumn));
        secondColumn ++;
      }
    }

    System.out.println("Calculated concatenated matrix:");
    printMatrix((Array2DRowRealMatrix) resMatrix);
    return resMatrix;
  }

  public static RealMatrix performReLuActivation(RealMatrix feededMatrix) {
    RealMatrix resMatrix = new Array2DRowRealMatrix(feededMatrix.getRowDimension(), feededMatrix.getColumnDimension());
    for (int row = 0; row < feededMatrix.getRowDimension(); row++) {
      for (int column = 0; column < feededMatrix.getColumnDimension(); column++) {
        if(feededMatrix.getEntry(row, column) > 0.0) {
          resMatrix.setEntry(row, column, feededMatrix.getEntry(row, column));
        }
      }
    }
    return resMatrix;
  }

  public static RealMatrix performNormalization(RealMatrix matrix, double epsilon) {
    int rows = matrix.getRowDimension();
    int cols = matrix.getColumnDimension();
    RealMatrix normalizedMatrix = new Array2DRowRealMatrix(rows, cols);
    for (int row = 0; row < rows; row++) {
      double sumSquares = 0;
      for (int column = 0; column < cols; column++) {
        double v = matrix.getEntry(row, column);
        sumSquares += v * v;
      }
      double rms = Math.sqrt(sumSquares / cols + epsilon);
      for (int column = 0; column < cols; column++) {
        normalizedMatrix.setEntry(row, column, matrix.getEntry(row, column) / rms);
      }
    }
    return normalizedMatrix;
  }

  public static double[] addVectors(double[] vec1, double[] vec2) {
    if(vec1.length != vec2.length) {
      throw new IllegalArgumentException("vectors of different lengths can not be added");
    }
    double[] result = new double[vec1.length];
    for (int i = 0; i < vec1.length; i++) {
      result[i] = vec1[i] + vec2[i];
    }
    return result;
  }
}
