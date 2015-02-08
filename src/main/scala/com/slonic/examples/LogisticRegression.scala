package com.slonic.examples

import breeze.linalg.sum
import com.slonic.LinearModel.LogisticRegression
import com.slonic.datasets.Iris
import com.slonic.Metrics._

object LogisticRegression extends App {
  println("Logistic regression example")
  val predLabel = "Iris-versicolor"

  val (labels, train) =  Iris.load()
  val y = labels.map(l => if(l == predLabel) 1.0 else 0.0)

  val clf = new LogisticRegression()
  clf.fit(train, y)

  val yPred = clf.predictProba(train)
  val err = sum(logloss(y, yPred)) / y.length
  println(s"Logloss : $err")
}
