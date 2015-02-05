package com.slonic.examples

import breeze.linalg.sum
import breeze.numerics.pow
import com.slonic.LinearModel.{LinearRegression, SGDRegressor}
import com.slonic.datasets.Iris

object LinearRegression extends App {
  println("Linear regression example")

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) / 1000000 + "ms")
    result
  }

  val (labels, train) =  Iris.load()
  val clf = new LinearRegression()
  val X = train(::, 1 to 3)

  // We will be learning to predict sepal length
  val y = train(::, 0)

  println("Fitting using closed form solution")
  time { clf.fit(X, y) }

  var yPred = clf.predict(X)

  var error = sum(pow((yPred - y), 2)) / y.length
  println(s"Standard Error: $error")

  println("Fitting using Gradient descent")
  clf.resetEstimator
  time { clf.fit(X, y, gradDescent = true) }

  yPred = clf.predict(X)

  error = sum(pow((yPred - y), 2)) / y.length
  println(s"Standard Error: $error")


  println("Fitting using Gradient descent (non vectorized solution)")
  clf.resetEstimator
  time { clf.fitGradientDescentNonVec(X, y) }

  yPred = clf.predict(X)

  error = sum(pow((yPred - y), 2)) / y.length
  println(s"Standard Error: $error")

  println("Fitting using Stochastic Gradient descent")
  val sgdClf = new SGDRegressor()
  time { sgdClf.fit(X, y) }

  yPred = sgdClf.predict(X)

  error = sum(pow((yPred - y), 2)) / y.length
  println(s"Standard Error: $error")
}
