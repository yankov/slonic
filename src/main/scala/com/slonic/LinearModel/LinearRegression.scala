package com.slonic.LinearModel

import breeze.linalg.{sum, inv, DenseVector, DenseMatrix}
import breeze.numerics.{abs, pow}
import com.slonic.base.BaseEstimator

// βˆ = (X.T * X)^-1  * X.T * y,
class LinearRegression extends BaseEstimator {

  var beta1: DenseVector[Double] = DenseVector[Double]()

  def resetEstimator = beta1 = DenseVector[Double]()

  def fit(train: DenseMatrix[Double], y: DenseVector[Double],
          gradDescent: Boolean = false, alpha: Double = 0.07,
          ep: Double = 0.000001): Unit = {
    if(gradDescent)
      fitGradientDescent(train, y)
    else
      fitMatrix(train, y)
  }

  /**
   * Below are 3 ways to fit the model for Linear regression.
   * fitMatrix - is a closed form solution. It is simple and fast,
   * but may perform badly if you have giant matrixes.
   *
   * fitGradientDescent is a way to solve it using gradient descent (duh)
   *
   * fitGradientDescentNonVec - same thing, but to calculate thetas insteas of
   * matrix/vectors operations I used simple loops. That is just maybe slightly easy
   * to understand if you don't remember how matrix multiplications works.
   *
   */

  // Closed-form solution. The fastest and the simplest.
  def fitMatrix(train: DenseMatrix[Double], y: DenseVector[Double]): Unit = {
    val b1 = DenseMatrix.ones[Double](y.length, 1)
    val trainI = DenseMatrix.horzcat(b1, train)
    beta1 = inv(trainI.t * trainI) * (trainI.t * y)
  }

  // Batch gradient descent solution. Maybe used when training set is too big.
  def fitGradientDescent(train: DenseMatrix[Double], y: DenseVector[Double],
                         alpha: Double = 0.007, ep: Double = 0.000001,
                         maxIter: Int = 1000000): Unit = {

    var thetha = DenseVector.zeros[Double](train.cols + 1)
    val t0 = DenseMatrix.ones[Double](y.length, 1)
    val trainI = DenseMatrix.horzcat(t0, train)

    val m = train.rows
    var err = (trainI * thetha) - y
    var e = 10.0
    var J = 100.0
    var nIter = 0

    while(abs(J - e) > ep && nIter <= maxIter) {
      nIter += 1
      J = e
      thetha = thetha :- ((trainI.t * err) :* 1.0/m) * alpha
      err = (trainI * thetha) - y
      e = sum(pow(err, 2))
    }

    println(s"Converged in $nIter iterations")
    beta1 = thetha
  }

  // Non vectorized solution for gradient descent.
  // Just in case you get lost with all matrix vectors multiplications
  def fitGradientDescentNonVec(train: DenseMatrix[Double], y: DenseVector[Double],
                         alpha: Double = 0.007, ep: Double = 0.000001,
                         maxIter: Int = 1000000): Unit = {

    var t0 = DenseMatrix.ones[Double](y.length, 1)
    var t1 = DenseVector.zeros[Double](train.cols + 1)

    // Add column for intercept
    val trainI = DenseMatrix.horzcat(t0, train)

    val m = train.rows

    // Still multiplying matrix and vector here to calculate error
    var err = (trainI * t1) - y
    var e = 10.0
    var J = 100.0
    var nIter = 0

    while(abs(J - e) > ep && nIter <= maxIter) {
      nIter += 1
      J = e
      for (i <- 0 to m - 1) {
        for (j <- 0 to trainI.cols - 1) {
          t1(j) = t1(j) - alpha * 1.0 / m * (err(i) * trainI(i, j))
        }
      }

      err = (trainI * t1) - y
      e = sum(pow(err, 2))
    }
    println(s"Converged in $nIter iterations")
    beta1 = t1
  }

  def predict(test: DenseMatrix[Double]): DenseVector[Double] = {
    val b1 = DenseMatrix.ones[Double](test.rows, 1)
    val testI = DenseMatrix.horzcat(b1, test)
    testI * beta1
  }

}
