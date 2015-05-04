package com.slonic.LinearModel

import breeze.linalg.{sum, inv, DenseVector, DenseMatrix}
import breeze.numerics.{abs, pow}
import com.slonic.base.BaseEstimator

class LinearRegression extends BaseEstimator {

  var beta1: DenseVector[Double] = DenseVector[Double]()

  def resetEstimator = beta1 = DenseVector[Double]()

  def fit(train: DenseMatrix[Double], y: DenseVector[Double],
          gradDescent: Boolean = false, C: Double = 1.0, alpha: Double = 0.01,
          ep: Double = 0.0001): Unit = {
    if(gradDescent)
      fitGradientDescent(train, y, alpha, ep, C)
    else
      fitMatrix(train, y, C)
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
  def fitMatrix(train: DenseMatrix[Double], y: DenseVector[Double], C: Double=1.0): Unit = {
    val b1 = DenseMatrix.ones[Double](y.length, 1)
    val trainI = DenseMatrix.horzcat(b1, train)
    val eye = DenseMatrix.eye[Double](trainI.cols)
    val lambda = 1 / C
    val reg = eye * lambda

    beta1 = inv(trainI.t * trainI + reg) * (trainI.t * y)
  }

  def Jcost(x: DenseMatrix[Double], theta: DenseVector[Double], y: DenseVector[Double] ) = {
    val m = x.rows
    var err = (x * theta) - y
    sum(pow(err, 2)) / (2*m.toDouble)
  }

  // Calculate gradient numerically for checking
  def gradNumerical(x: DenseMatrix[Double], theta: DenseVector[Double], y: DenseVector[Double],
                    ep: Double = 0.0001): Double = {
    (Jcost(x, theta + ep, y) - Jcost(x, theta - ep, y)) / (2 * ep)
  }

  // Batch gradient descent solution. Maybe used when training set is too big.
  def fitGradientDescent(train: DenseMatrix[Double], y: DenseVector[Double],
                         alpha: Double = 0.01, ep: Double = 0.0001,
                         C: Double = 1.0, maxIter: Int = 1000000): Unit = {

    val t0 = DenseMatrix.ones[Double](y.length, 1)
    var t1 = DenseVector.zeros[Double](train.cols + 1)
    val trainI = DenseMatrix.horzcat(t0, train)

    val m = train.rows
    var err = (trainI * t1) - y
    var e = 10.0
    var J = 100.0
    var nIter = 0
    val lambda = 1.0 / C

    while(abs(J - e) > ep && nIter <= maxIter) {
      nIter += 1
      J = e
      val grad = (trainI.t * err) / m.toDouble
//      println("Gradient")
//      println(sum(grad))
//      println("Numerical gradient")
//      println(gradNumerical(trainI, t1, y))
      t1 = t1 :- (grad :+ t1 * (lambda / m)) * alpha
      err = (trainI * t1) - y

      e = (sum(pow(err, 2)) + lambda * sum(t1 * t1.t)) / (2*m)
    }

    println(s"Converged in $nIter iterations")
    beta1 = t1
  }

  // Non vectorized solution for gradient descent.
  // Just in case you get lost with all matrix vectors multiplications
  def fitGradientDescentNonVec(train: DenseMatrix[Double], y: DenseVector[Double],
                         alpha: Double = 0.01, ep: Double = 0.0001, C: Double = 1.0,
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
    val lambda = 1.0 / C
    var grad = 0.0

    while(abs(J - e) > ep && nIter <= maxIter) {
      nIter += 1
      J = e
      for (j <- 0 to trainI.cols - 1) {
        grad = 0.0
        for (i <- 0 to m - 1) {
          grad += err(i) * trainI(i, j)
        }
        t1(j) = t1(j) - alpha * ((grad / m) + t1(j) * (lambda / m))
      }

      err = (trainI * t1) - y
      e = (sum(pow(err, 2)) + lambda * sum(t1 * t1.t)) / (2*m)
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
