package com.slonic.datasets

import java.io.{BufferedInputStream, InputStream}
import java.util.zip.GZIPInputStream

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.io.Source

/**
 * This dataset was taken from coursera machine learning course.
 * I think it is a sample from MNIST that has been already normalized.
 */
object Digits5k {
  def gis(s: InputStream) = new GZIPInputStream(new BufferedInputStream(s))

  def load(): (DenseVector[Int], DenseMatrix[Double]) = {
    val f1 = Source.fromInputStream(gis(getClass.getResourceAsStream("/datasets/digits_train.csv.gz")))
    val train = f1.getLines().toList.map(_.split(",").map(_.toDouble))

    val f2 = Source.fromInputStream(gis(getClass.getResourceAsStream("/datasets/digits_labels.csv.gz")))
    val labels = f2.getLines().toList.map(_.toInt)

    (DenseVector(labels:_*), DenseMatrix(train:_*))
  }

}
