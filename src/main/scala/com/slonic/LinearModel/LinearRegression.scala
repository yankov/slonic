package com.slonic.LinearModel

import breeze.linalg.{inv, DenseVector, DenseMatrix}
import com.slonic.base.BaseEstimator

// βˆ = (X.T * X)^-1  * X.T * y,
class LinearRegression extends BaseEstimator {

  var beta1: DenseVector[Double] = DenseVector[Double]()

  def fit(train: DenseMatrix[Double], y: DenseVector[Double]): Unit = {
    beta1 = inv(train.t * train) * (train.t * y)
  }

  def predict(test: DenseMatrix[Double]): DenseVector[Double] = {
    test * beta1
  }

}
