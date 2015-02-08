package com.slonic.LinearModel

import breeze.linalg.{sum, DenseMatrix, DenseVector}
import breeze.numerics.{pow, abs}
import com.slonic.base.BaseEstimator
import scala.util.control.Breaks._

/**
 * Linear model fitted by minimizing a loss with stochastic
 * gradient descent. This models is good for big data sets
 * since it can be trained on small batches of data (even if you feed it
 * one sample at a time).
 */
class SGDRegressor extends BaseEstimator {

  var beta1: DenseVector[Double] = DenseVector[Double]()

  def fit(train: DenseMatrix[Double], y: DenseVector[Double],
          alpha: Double = 0.007, ep: Double = 0.000001,
          maxIter: Int = 1000000): Unit = {

    var t1 = DenseVector.zeros[Double](train.cols + 1)
    val t0 = DenseMatrix.ones[Double](y.length, 1)
    val trainI = DenseMatrix.horzcat(t0, train)

    val m = train.rows
    var err = (trainI * t1) - y
    var e = 10.0
    var J = 100.0
    var nIter = 0

    tryBreakable {
      while(nIter <= maxIter) {
        nIter += 1
        J = e

        for (i <- 0 to m - 1) {
          for (j <- 0 to trainI.cols - 1) {
            t1(j) = t1(j) - alpha * (err(i) * trainI(i, j))
          }
          err = (trainI * t1) - y
          e = sum(pow(err, 2))
          if (abs(J - e) <= ep) break
        }
      }
    } catchBreak { println(s"Converged in $nIter iterations") }

    beta1 = t1
  }

  def predict(test: DenseMatrix[Double]): DenseVector[Double] = {
    val b1 = DenseMatrix.ones[Double](test.rows, 1)
    val testI = DenseMatrix.horzcat(b1, test)
    testI * beta1
  }
}
