package com.slonic.datasets

import java.io.{InputStream, BufferedInputStream}
import java.util.zip.GZIPInputStream

import breeze.linalg.{DenseMatrix, DenseVector}
import scala.io.Source

object Digits {
  val NUM_FEATURES = 64
  def gis(s: InputStream) = new GZIPInputStream(new BufferedInputStream(s))

  def load(): (DenseVector[Int], DenseMatrix[Double])  = {
    val src = Source.fromInputStream(gis(getClass.getResourceAsStream("/datasets/digits.csv.gz")))

    val rows = src.getLines().map(_.split(",")).toList.map(row => (row.last.toInt, row.take(NUM_FEATURES).map(_.toDouble)))

    (DenseVector(rows.map(_._1):_*), DenseMatrix(rows.map(_._2):_*))
  }
}
