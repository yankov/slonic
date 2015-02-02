package com.slonic.LinearModel

import breeze.linalg.{inv, DenseVector, DenseMatrix}
import com.slonic.base.BaseEstimator

// βˆ = (X.T * X)^-1  * X.T * y,
class LinearRegression extends BaseEstimator {

  val featureNames = (
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)")

  var beta1: DenseVector[Double] = DenseVector[Double]()

  def fit(train: DenseMatrix[Double], y: DenseVector[Double]): Unit = {
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
