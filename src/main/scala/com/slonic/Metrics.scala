package com.slonic

import breeze.linalg.{DenseVector, sum}
import breeze.numerics.log

object Metrics {

  def logloss(y: DenseVector[Double], yPred: DenseVector[Double]) = {
    val ones = DenseVector.ones[Double](y.length)
    y :* log(yPred) :+ (ones - y) :* log(ones - yPred)
  }

}
