package com.slonic.LinearModel

import breeze.linalg.{sum, DenseMatrix, DenseVector}
import breeze.numerics._
import com.slonic.Metrics._

class LogisticRegression {
  var beta1: DenseVector[Double] = DenseVector[Double]()

  val EXP_MAX = 20.0
  val EXP_MIN = -20.0

  def z(t1: DenseVector[Double], x: DenseMatrix[Double]): DenseVector[Double] = x * t1

  def h(t1: DenseVector[Double], x: DenseMatrix[Double]): DenseVector[Double] = {
    val ones = DenseVector.ones[Double](x.rows)

    // Prevent overflow and underflow of exp cause result can fall outside of the Double precision range
    val z0 = -z(t1, x).map( v =>
     if (v > EXP_MAX) EXP_MAX
     else if(v < EXP_MIN) EXP_MIN
     else v
    )

    ones / (exp(z0) + 1.0)
  }

  // Looks very similar to stochastic linear regressor, except for logloss and hypothesis function
  def fit(train: DenseMatrix[Double], y: DenseVector[Double],
          alpha: Double = .001, ep: Double = 0.0001,
          maxIter: Int = 100000): Unit = {

    var t1 = DenseVector.zeros[Double](train.cols + 1)
    val t0 = DenseMatrix.ones[Double](y.length, 1)
    val trainI = DenseMatrix.horzcat(t0, train)

    val m = train.rows
    var yPred = h(t1, trainI)
    var err = yPred - y
    var e = -1.0/m * sum(logloss(y, yPred))
    var J = 100.0
    var nIter = 0

    while(abs(J - e) > ep && nIter <= maxIter) {
      nIter += 1
      J = e
      t1 = t1 :- ((trainI.t * err) * alpha)
      yPred = h(t1, trainI)
      err =  yPred - y
      e = -1.0/m * sum(logloss(y, yPred))
    }

    println(s"Converged in $nIter iterations")
    beta1 = t1
  }

  def predictProba(test: DenseMatrix[Double]): DenseVector[Double] = {
    val b1 = DenseMatrix.ones[Double](test.rows, 1)
    val testI = DenseMatrix.horzcat(b1, test)
    h(beta1, testI)
  }

  def predict(test: DenseMatrix[Double]): DenseVector[Double] = {
    val yPred = predictProba(test)
    yPred.map(x => if(x > .5) 1.0 else .0)
  }

}
