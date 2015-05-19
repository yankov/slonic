package com.slonic.examples

import breeze.linalg.{DenseVector, DenseMatrix}
import com.slonic.NeuralNetwork.{ANN, Perceptron}
import com.slonic.datasets.Digits

object NeuralNetworks extends App {
  println("Perceptron example")
  val predLabel = 9
  val (labels, train) = Digits.load()
  val y = labels.map(l => if(l == predLabel) 1 else 0)

//  val perceptron = new Perceptron()
//  perceptron.fit(train, y)
//  val yPred = perceptron.predict(train)
//  println(s"Weights: ${perceptron.weights}")

  val nn = new ANN(inputSize = train.cols, nHidden = 2)
  val ones = DenseVector.ones[Double](train.rows)
  val H = DenseMatrix.eye[Double](train.rows) - (ones * ones.t)
  val trainC = H * train
  nn.fit(train, labels, checkGradients = true)
  //val yPred = nn.predict(train)
}
