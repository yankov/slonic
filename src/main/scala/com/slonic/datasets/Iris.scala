package com.slonic.datasets

import breeze.linalg._
import breeze.numerics._
import scala.io.Source
import scala.util.Try

object Iris {

  def getDouble(v: => String, default: Double = .0): Double = Try(v.toDouble).getOrElse(default)
  def getString(v: => String, default: String = "NA"): String = Try(v).getOrElse(default)

  def load(): (DenseVector[String], DenseMatrix[Double])  = {
    val src = Source.fromURL(getClass.getResource("/datasets/iris.csv"))
    val iter = src.getLines().map(_.split(","))
    val features = iter.map(l =>
      (getString(l(4)), (getDouble(l(0)), getDouble(l(1)), getDouble(l(2)), getDouble(l(3))))
    ).toList

    (DenseVector(features.map(_._1):_*), DenseMatrix(features.map(_._2):_*))
  }
}
