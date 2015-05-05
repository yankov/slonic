package com.slonic.NeuralNetwork

import breeze.linalg.{DenseMatrix, DenseVector}

class Perceptron(alpha: Double = 0.1, threshold: Double = 0.5) {

  var weights: DenseVector[Double] = DenseVector[Double]()

  /*
   * Single-layer perceptron. Binary classification only.
   */
  def fit(train: DenseMatrix[Double], y: DenseVector[Int], maxIter: Int = 1000): Unit = {
    weights = DenseVector.zeros[Double](train.cols + 1)
    var iter = 1

    // add bias
    val b = DenseMatrix.ones[Double](y.length, 1)
    val trainI = DenseMatrix.horzcat(b, train)
    var errorCount = -1

    while(iter <= maxIter && errorCount != 0) {
      println(s"Iteration $iter, error count = $errorCount")
      errorCount = 0
      for(j <- 0 to trainI.rows - 1) {
        val yPred = if((trainI(j, ::) * weights) > threshold) 1 else 0
        val err = y(j) - yPred
        if (err != 0) {
          iter += 1
          errorCount += 1
          weights += (trainI(j, ::).inner * err.toDouble) * alpha
        }
      }
    }

    if(errorCount == 0) println(s"Converged after $iter iterations")
    else println(s"max_iter has been reached with errorCount = $errorCount")
  }

  def predict(test: DenseMatrix[Double]): DenseVector[Int] = {
    val b = DenseMatrix.ones[Double](test.rows, 1)
    val testI = DenseMatrix.horzcat(b, test)

    (testI * weights).map(y => if(y > threshold) 1 else 0)
  }


}
