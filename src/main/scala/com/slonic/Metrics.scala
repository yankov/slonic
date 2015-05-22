package com.slonic

import breeze.linalg.{DenseVector, sum}
import breeze.numerics.{pow, log}

object Metrics {

  def logloss(y: DenseVector[Double], yPred: DenseVector[Double]) = {
    val ones = DenseVector.ones[Double](y.length)
    y :* log(yPred) :+ (ones - y) :* log(ones - yPred)
  }

  // Mean squared error
  def mse(y: DenseVector[Double], yPred: DenseVector[Double]): Double =
    sum(pow(y - yPred, 2)) / y.length.toDouble

  // Accuracy of classification
  def accuracy(y: DenseVector[Int], yPred: DenseVector[Int]): Double =
    // a simpler solution should have worked, but elementwise comparison
    // works strangely in Breeze a :== b will return a BitVector[Int]
    y.toArray.zipWithIndex.filter { case (e, i) => e == yPred(i) }.length / y.length.toDouble
}
