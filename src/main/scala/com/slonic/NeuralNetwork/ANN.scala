package com.slonic.NeuralNetwork

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Rand

import scala.annotation.tailrec

class ANN(inputSize: Int, nHidden: Int = 2, nOutput: Int = 10, alpha: Double = 0.3,
          ep: Double = 0.000001, maxIter: Int = 10000, C: Double = 1) {

  /* Constants for gradient checking */
  // Layer to check (running on all layers will take too long)
  val LAYER_CHK = 0
  // Number of neurons to check
  val NUM_NODES_CHK = 3
  // Number of weights to check (all of them by default)
  val NUM_WEIGHTS_CHK = inputSize

  // Acceptable order of difference between gradients
  val EPSILON_CHK = 10e-7

  // Number of nodes in a hidden layer
  val LAYER_SIZE = 25

  val lambda = alpha
//  var weights = (0 to nHidden).map(x =>
//    if (x == 0)
//      DenseMatrix.rand[Double](LAYER_SIZE, inputSize + 1, rand = Rand.gaussian) //* (2 * .12)) - .12
//    else if (x < nHidden)
//      DenseMatrix.rand[Double](LAYER_SIZE, LAYER_SIZE + 1, rand = Rand.gaussian) //* (2 * .12)) - .12
//    else
//      DenseMatrix.rand[Double](nOutput, LAYER_SIZE + 1, rand = Rand.gaussian) //* (2 * .12)) - .12
//  ).toArray
//
//  var D = (0 to nHidden).map(x =>
//    if (x == 0)
//      DenseMatrix.zeros[Double](LAYER_SIZE, inputSize + 1)
//    else if(x < nHidden)
//      DenseMatrix.zeros[Double](LAYER_SIZE, LAYER_SIZE + 1)
//    else
//      DenseMatrix.zeros[Double](nOutput, LAYER_SIZE + 1)
//  ).toArray

  // values for 'a' for different layers during forward-propagation
  var aVec: List[DenseVector[Double]] = List()

  def sigmoid(z: DenseVector[Double]): DenseVector[Double] = {
    val ones = DenseVector.ones[Double](z.length)
    ones / (exp(z) + 1.0)
  }

  def inputLayer: DenseMatrix[Double] =
    DenseMatrix.zeros[Double](LAYER_SIZE, inputSize + 1)

  def hiddenLayer: DenseMatrix[Double] =
    DenseMatrix.zeros[Double](LAYER_SIZE, LAYER_SIZE + 1)

  def hiddenLayers(rand: Boolean = false): Array[DenseMatrix[Double]] =
    (for(_ <- 0 to nHidden - 1) yield if(rand) randInit(hiddenLayer) else hiddenLayer).toArray

  def outputLayer =
    DenseMatrix.zeros[Double](nOutput, LAYER_SIZE + 1)

  def randInit(x: DenseMatrix[Double]) =
    DenseMatrix.rand[Double](x.rows, x.cols, rand = Rand.gaussian)


  val weights: Array[DenseMatrix[Double]] =
    randInit(inputLayer) +: hiddenLayers(rand = true) +: randInit(outputLayer) +: Array()

  val D: Array[DenseMatrix[Double]] =
    inputLayer +: hiddenLayers() +: outputLayer +: Array()

  // Forward propagation
  def h(x: DenseVector[Double], thetas: Array[DenseMatrix[Double]]): DenseVector[Double] = {
    var a = x
    var z = DenseVector[Double]()
    aVec = List(a)

    for (l <- 0 to thetas.length - 1) {
      z = thetas(l) * a
      a = sigmoid(z)

      if(l < thetas.length - 1) {
        a = DenseVector(1.0 +: a.toArray) // add a bias unit
        aVec = aVec :+ a
      }
    }

    a
  }

  // Forward propagation
//  @tailrec
//  final def h(x: DenseVector[Double], thetas: Array[DenseMatrix[Double]]): DenseVector[Double] = {
//   if(thetas.isEmpty)
//      x(1 to -1)
//    else {
//      aVec = aVec :+ x
//      val a = sigmoid(thetas.head * x).toArray
//      h(DenseVector(1.0 +: a), thetas.tail)
//    }
//  }

  def binarize(ys: DenseVector[Int]): DenseMatrix[Double] = {
    val kl = ys.toArray.toSet
    val ysBin = DenseMatrix.zeros[Double](ys.length, kl.size)
    ys.data.zipWithIndex.foreach{case (y, i) => ysBin.update(i, y - 1, 1.0)}
    ysBin
  }

  def J(x: DenseMatrix[Double], thetas: Array[DenseMatrix[Double]], y: DenseMatrix[Double], regularize: Boolean = true): Double = {
    val ones = DenseVector.ones[Double](y.cols)
    val m = x.rows

    val j = -1/m.toDouble * (0 to m - 1).foldLeft(0.0) {(acc, i) =>
      val hyp = h(x(i, ::).inner, thetas)
      val yi = y(i,::).inner
      val v = (yi :* log(hyp)) :+ ((ones - yi) :* log(ones - hyp))
      acc + sum(v)
    }

    if(regularize) {
      j //+ thetas.map(t => sum(sum(t :* t, Axis._0))).sum * (lambda / (2 * m.toDouble))
    } else
      j
  }

  def sigmoidGrad(a: DenseVector[Double]): DenseVector[Double] = {
    val ones = DenseVector.ones[Double](a.length)
    a :* (ones - a)
  }


  def fit(train: DenseMatrix[Double], y: DenseVector[Int], maxIter: Int = 100000,
          checkGradients: Boolean = false): Unit = {

    val ys = binarize(y)
    val bias = DenseMatrix.ones[Double](train.rows, 1)
    val trainI = DenseMatrix.horzcat(bias, train)

    var cost = J(trainI, weights, ys)
    val m = trainI.rows
    var e = 100.0
    var nIter = 0

    // Backpropagation
    println(s"Initial cost $cost")
    while(nIter <= maxIter) {
      e = cost
      for (i <- 0 to m - 1) {
        var a = h(trainI(i, ::).inner, weights) // a4
        var delta = a - ys(i, ::).inner // delta4

        for (l <- weights.length - 1 to 0 by - 1) {
          a = aVec(l)
          val t0 = D(l)(::, 0)
          D(l) = D(l) + (delta * a.t) // delta(l+1) * a.t(l)
          D(l)(::, 0) := t0 :* delta  // bias deltas

          a = a.slice(1, a.length)
          delta = (weights(l).t(1 to -1, ::) * delta) :* sigmoidGrad(a) // delta(l) = theta(l) * delta(l+1) * a'
        }

   /*
        a = aVec(1, ::).inner // a3
        D(2) = D(2) + (delta * a.t) // delta4 * a3

        da = a :* (ones - a) // a3'
        delta = (weights(2).t * delta) :* da // delta3 = Theta3 * delta4 :* a3'

        a = aVec(0, ::).inner // a2
        D(1) = D(1) + (delta * a.t) // delta3 * a2

        da = a :* (ones - a) // a2'
        delta = (weights(1).t * delta) :* da // delta2 = Theta2 * delta3 :* a2'

        a = trainI(i, ::).inner // a1
        D(0) = D(0) + (delta * a.t)
   */

      }

      D = D.map(_ :/ m.toDouble)

      // After the first run, compare values with numerically calculated gradients to make sure
      // there are no errors in backprop implementation
      if(nIter == 0 && checkGradients) {
        println(s"Running gradient check for L=$LAYER_CHK")
        val grOk = areGradsOk(trainI, ys, LAYER_CHK, NUM_NODES_CHK, NUM_WEIGHTS_CHK)
        if(grOk)
          println("Gradients check: PASSED!")
        else
          println("!!! Gradients check: FAILED!")
      }

      // Update weights
      for (l <- 0 to weights.length - 1) {
        weights(l)(::, 0) := D(l)(::, 0)
        weights(l)(::, 1 to -1) := weights(l)(::, 1 to -1) + D(l)(::, 1 to -1) * alpha
      }


      cost = J(trainI, weights, ys)
      println(s"Cost $cost at epoch $nIter")
      nIter += 1
    }

  }


  // Check gradients for a given layer, nodes and weights
  def areGradsOk(trainI: DenseMatrix[Double], ys: DenseMatrix[Double], ln: Int, rows: Int, cols: Int): Boolean = {
    val numGrads = numGrad(trainI, ys, ln, (0 to rows).toList, (0 to cols).toList)
    val diff = abs(D(ln)(0 to rows, 0 to cols).t.toDenseVector) - abs(DenseVector(numGrads:_*))
    diff.forall(v => v < EPSILON_CHK)
  }

  // Calculate gradients numerically
  def numGrad(trainI: DenseMatrix[Double], ys: DenseMatrix[Double], l: Int, rows: List[Int], cols: List[Int]): List[Double] = {
    val ws = weights.map(_.copy)

    for {
        row <- rows
        col <- cols
        _ = ws(l).update(row, col, ws(l)(row, col) + ep)
        j1 = J(trainI, ws, ys, regularize = false)

        _ = ws(l).update(row, col, ws(l)(row, col) - 2 * ep)
        j2 = J(trainI, ws, ys, regularize = false)

        _ = ws(l).update(row, col, ws(l)(row, col) + ep) // restoring the value
    } yield (j1 - j2) / (2 * ep)

  }

  //def predict(test: DenseMatrix[Double]): DenseMatrix[Double] = h(test, weights)

}
