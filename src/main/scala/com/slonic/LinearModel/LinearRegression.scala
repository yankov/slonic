package com.slonic.LinearModel

import breeze.linalg.{sum, inv, DenseVector, DenseMatrix}
import breeze.numerics.{abs, pow}
import com.slonic.base.BaseEstimator

// βˆ = (X.T * X)^-1  * X.T * y,
class LinearRegression extends BaseEstimator {

  var beta1: DenseVector[Double] = DenseVector[Double]()

  def fit(train: DenseMatrix[Double], y: DenseVector[Double],
          gradDescent: Boolean = false, alpha: Double = 0.07): Unit = {
    if(gradDescent)
      fitGradientDescent(train, y)
    else
      fitMatrix(train, y)
  }

  // Batch gradient descent solution
  def fitGradientDescent(train: DenseMatrix[Double], y: DenseVector[Double],
                         alpha: Double = 0.007, ep: Double = 0.0001): Unit = {

    var thetha = DenseVector.zeros[Double](train.cols + 1)
    val t0 = DenseMatrix.ones[Double](y.length, 1)
    val trainI = DenseMatrix.horzcat(t0, train)

    val m = train.rows
    var err = (trainI * thetha) - y
    var e = 10.0
    var J = 100.0
    var n_iter = 0

    while(abs(J - e) > ep) {
      n_iter += 1
      J = e
      thetha = thetha :- ((trainI.t * err) :* 1.0/m) * alpha
      err = (trainI * thetha) - y
      e = sum(pow(err, 2))
    }

    println(s"Converged in $n_iter iterations")
    beta1 = thetha
  }

  // Closed-form solution
  def fitMatrix(train: DenseMatrix[Double], y: DenseVector[Double]): Unit = {
    val b1 = DenseMatrix.ones[Double](y.length, 1)
    val trainI = DenseMatrix.horzcat(b1, train)
    beta1 = inv(trainI.t * trainI) * (trainI.t * y)
  }

  def predict(test: DenseMatrix[Double]): DenseVector[Double] = {
    val b1 = DenseMatrix.ones[Double](test.rows, 1)
    val testI = DenseMatrix.horzcat(b1, test)
    testI * beta1
  }

}
