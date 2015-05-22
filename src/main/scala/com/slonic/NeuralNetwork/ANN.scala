package com.slonic.NeuralNetwork

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Rand
import scala.annotation.tailrec
import java.io.File
import com.slonic.Metrics.accuracy

class ANN(name: String, train: DenseMatrix[Double], labels: DenseVector[Int], nHidden: Int = 2,
          layerSize: Int = 25, ep: Double = 10e-5) {

  // Size of a training vector
  val inputSize = train.cols

  // Number of labels (size of the output vector)
  val outputSize = labels.toArray.toSet.size

  // Convert labels to a matrix of indicator variables
  val y = binarize(labels)

  // Add bias vector
  val trainI = DenseMatrix.horzcat(DenseMatrix.ones[Double](train.rows, 1), train)

  /* Constants for gradient checking */
  // Layer to check (running on all layers will take too long)
  val LAYER_CHK = 0

  // Number of neurons to check
  val NUM_NODES_CHK = 3

  // Number of weights to check (all of them by default)
  val NUM_WEIGHTS_CHK = 60

  // Acceptable order of difference between gradients
  val EPSILON_CHK = 10e-7

  // Weights are initialized from random gaussian with mean 0 and std 1
  var weights: Array[DenseMatrix[Double]] = (0 to nHidden).map(createLayer).toArray

  def createLayer(n: Int): DenseMatrix[Double] = n match {
    // first hidden layer
    case 0 => DenseMatrix.rand[Double](layerSize, inputSize + 1, rand = Rand.gaussian)
    // output layer
    case `nHidden` => DenseMatrix.rand[Double](outputSize, layerSize + 1, rand = Rand.gaussian)
    // other hidden layers
    case _ => DenseMatrix.rand[Double](layerSize, layerSize + 1, rand = Rand.gaussian)
  }

  def sigmoidGrad(a: DenseVector[Double]): DenseVector[Double] = a :* (-a + 1.0)

  def zeroCopy(mx: Array[DenseMatrix[Double]]) =
    mx.map(m => DenseMatrix.zeros[Double](m.rows, m.cols))

  def binarize(ys: DenseVector[Int]): DenseMatrix[Double] = {
    val ysBin = DenseMatrix.zeros[Double](ys.length, outputSize)
    ys.data.zipWithIndex.foreach{case (y, i) => ysBin.update(i, y - 1, 1.0)}
    ysBin
  }

  def loss(y: DenseVector[Double], yPred: DenseVector[Double]): DenseVector[Double] =
    (-y :* log(yPred)) :- ((-y + 1.0) :* log(-yPred + 1.0))

  // Cost function
  def J(x: DenseMatrix[Double], thetas: Array[DenseMatrix[Double]], y: DenseMatrix[Double], regularize: Boolean = true,
        lambda: Double = 0.8): Double = {
    val j = (1.0/x.rows) * sum(for {
      i <- 0 to x.rows - 1
      yPred = h(x(i, ::).inner, thetas)._1
      yi = y(i, ::).inner
    } yield sum(loss(yi, yPred)))

    if(regularize) {
      j + thetas.map(t => sum(sum(pow(t(::, 1 to -1), 2), Axis._0))).sum * (lambda / (2.0 * x.rows))
    } else
      j
  }

  // Forward propagation for a single training vector
  @tailrec
  final def h(x: DenseVector[Double], thetas: Array[DenseMatrix[Double]],
              aVec: List[DenseVector[Double]] = List()): (DenseVector[Double], List[DenseVector[Double]])= {
    if(thetas.isEmpty)
      (x(1 to -1), aVec)
    else {
      val a = sigmoid(thetas.head * x).toArray
      h(DenseVector(1.0 +: a), thetas.tail, aVec :+ x(1 to -1))
    }
  }

  // Forward propagation for the entire training set
  @tailrec
  final def h(x: DenseMatrix[Double], thetas: Array[DenseMatrix[Double]]): DenseMatrix[Double]= {
   if(thetas.isEmpty)
      x(1 to -1, ::).t
    else {
      var a: DenseMatrix[Double] = if(thetas.length > 1)
        sigmoid(thetas.head * x.t)
       else {
        sigmoid(thetas.head * x)
      }
      a = DenseMatrix.vertcat[Double](DenseMatrix.ones[Double](1, a.cols), a)
      h(a, thetas.tail)
    }
  }

  // Backpropagation. Returns matrix of gradients.
  def backpropagate(train: DenseMatrix[Double], ys: DenseMatrix[Double],
                    weights: Array[DenseMatrix[Double]]): Array[DenseMatrix[Double]] = {

    val D = zeroCopy(weights)
    D.foreach(d => d(::, 0) := 1.0)

    for (i <- 0 to train.rows - 1) {
      var (a, aVec) = h(train(i, ::).inner, weights) // get predictions and activation for every layer
      var delta = a - ys(i, ::).inner // delta(L)

      for (l <- weights.length - 1 to 0 by - 1) {
        a = aVec(l)
        D(l)(::, 0) := D(l)(::, 0) :* delta  // bias deltas
        D(l)(::, 1 to -1) := D(l)(::, 1 to -1) + (delta * a.t) // delta(l+1) * a.t(l)

        delta = (weights(l).t(1 to -1, ::) * delta) :* sigmoidGrad(a) // delta(l) = theta(l) * delta(l+1) * a'
      }
    }

    D.map(_ :/ train.rows.toDouble)
  }

  def train(maxIter: Int = 10000, checkGradients: Boolean = false, restoreSnapshot: Boolean = false,
            lambda: Double = 0.8): Unit = {

    if(restoreSnapshot)
      weights = readWeights(name)
    var cost = J(trainI, weights, y, lambda = lambda)
    var nIter = 0

    // Backpropagation
    println(s"Initial cost $cost")
    while(nIter <= maxIter) {

      val D = backpropagate(trainI, y, weights)

      // Update weights
      for (l <- 0 to weights.length - 1) {
        weights(l)(::, 0) := D(l)(::, 0)
        weights(l)(::, 1 to -1) := weights(l)(::, 1 to -1) - D(l)(::, 1 to -1) * lambda
      }

      // After the first run, compare values with numerically calculated gradients to make sure
      // there are no errors in backprop implementation
      if(nIter == 0 && checkGradients) {
        println(s"Running gradient check for L=$LAYER_CHK")
        val grOk = areGradsOk(D, LAYER_CHK, NUM_NODES_CHK, NUM_WEIGHTS_CHK)
        if(grOk)
          println("Gradients check: PASSED!")
        else {
          println("!!! Gradients check: FAILED!")
          sys.exit(1)
        }
      }

      cost = J(trainI, weights, y, lambda = lambda)
      println(s"Cost $cost at epoch $nIter")
      if((nIter + 1) % 100 == 0) {
        val yPred = predict(train)
        println("Accuracy: " + accuracy(labels, DenseVector(yPred:_*)))
        println(s"Saving the weights at epoch $nIter")
        saveWeights(weights)
      }

      nIter += 1
    }

    saveWeights(weights)
  }

  // Check gradients for a given layer, nodes and weights
  def areGradsOk(D: Array[DenseMatrix[Double]], ln: Int, rows: Int, cols: Int): Boolean = {
    val numGrads = numGrad(ln, (0 to rows).toList, (0 to cols).toList)
    val diff = abs(D(ln)(0 to rows, 0 to cols).t.toDenseVector) - abs(DenseVector(numGrads:_*))
    diff.forall(v => v < EPSILON_CHK)
  }

  // Calculate gradients numerically
  def numGrad(l: Int, rows: List[Int], cols: List[Int]): List[Double] = {
    val ws = weights.map(_.copy)

    for {
        row <- rows
        col <- cols
        _ = ws(l).update(row, col, ws(l)(row, col) + ep)
        j1 = J(trainI, ws, y, regularize = false)

        _ = ws(l).update(row, col, ws(l)(row, col) - 2 * ep)
        j2 = J(trainI, ws, y, regularize = false)

        _ = ws(l).update(row, col, ws(l)(row, col) + ep) // restoring the value
    } yield (j1 - j2) / (2 * ep)

  }

  def saveWeights(weights: Array[DenseMatrix[Double]]) =
    weights.zipWithIndex.foreach{case (ws, i) =>  csvwrite(new File(s"models/$name-theta${i+1}.csv"), ws)}

  def readWeights(name: String): Array[DenseMatrix[Double]] =
    (0 to nHidden).map(i => csvread(new File(s"models/$name-theta${i+1}.csv"))).toArray

  def loadSnapshot(name: String) =
    weights = readWeights(name)

  def predictProba(test: DenseMatrix[Double]): DenseMatrix[Double] = {
    val bias = DenseMatrix.ones[Double](test.rows, 1)
    val testI = DenseMatrix.horzcat(bias, test)

    h(testI, weights)
  }

  def predict(test: DenseMatrix[Double]): List[Int] = {
    val yPred = predictProba(test)
    val preds = for {
      i <- 0 to yPred.rows - 1
      y = yPred(i,::).inner
    } yield y.findAll(_ == max(y)).head

    preds.map(_ + 1).toList
  }

}
