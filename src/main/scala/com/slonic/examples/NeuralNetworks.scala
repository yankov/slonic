package com.slonic.examples

import com.slonic.NeuralNetwork.{ANN, Perceptron}
import com.slonic.datasets.{Digits5k, Digits}

object NeuralNetworks extends App {
  val (labels, train) = Digits5k.load()

//  println("Perceptron example")
//  val predLabel = 9
  //val y = labels.map(l => if(l == predLabel) 1 else 0)
//  val perceptron = new Perceptron()
//  perceptron.fit(train, y)
//  val yPred = perceptron.predict(train)
//  println(s"Weights: ${perceptron.weights}")

  println("Multi-layer neural net example")
  val nn = new ANN(inputSize = train.cols, nHidden = 1)

 // Substract mean
 // for(i <- 0 to train.rows - 1) {
 //   val v = train(i, ::).inner
 //   train(i, ::).inner := v - (sum(v) / v.length)
 // }

  nn.train(train, labels, checkGradients = false, maxIter = 5)
  //val yPred = nn.predictProba(train)
  //val yPred = nn.predict(train)
}
