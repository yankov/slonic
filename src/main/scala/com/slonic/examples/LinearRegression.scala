package com.slonic.examples

import breeze.linalg.sum
import breeze.numerics.pow
import com.slonic.LinearModel.LinearRegression
import com.slonic.datasets.Iris

object LinearRegression extends App {
  println("Linear regression example")

  val (labels, train) =  Iris.load()
  val clf = new LinearRegression()
  val X = train(::, 1 to 3)

  // will be training to predict sepal length
  val y = train(::, 0)
  clf.fit(X, y)

  val y_pred = clf.predict(X)

  val error = sum(pow((y_pred - y), 2)) / y.length
  println(s"Standard Error: $error")
}
