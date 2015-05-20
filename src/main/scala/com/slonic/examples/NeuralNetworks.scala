package com.slonic.examples

import breeze.linalg._
import com.slonic.NeuralNetwork.{ANN, Perceptron}
import com.slonic.datasets.{Digits5k, Digits}

object NeuralNetworks extends App {
  println("Perceptron example")
  val predLabel = 9
  val (labels, train) = Digits5k.load()
  //val y = labels.map(l => if(l == predLabel) 1 else 0)

//  val perceptron = new Perceptron()
//  perceptron.fit(train, y)
//  val yPred = perceptron.predict(train)
//  println(s"Weights: ${perceptron.weights}")

  // 1 layer 5k = 3.2 lost
  // 2 layer 5k = 2.8 lost (no reg)/ 2.32 at epoch 1000
  val nn = new ANN(inputSize = train.cols, nHidden = 1)

 // val ones = DenseVector.ones[Double](train.rows)
 // val H = DenseMatrix.eye[Double](train.rows) - (ones * ones.t)
 // val trainC = H * train

 // Substract mean
 // for(i <- 0 to train.rows - 1) {
 //   val v = train(i, ::).inner
 //   train(i, ::).inner := v - (sum(v) / v.length)
 // }

  nn.fit(train, labels, checkGradients = true)
  //val yPred = nn.predict(train)
}
