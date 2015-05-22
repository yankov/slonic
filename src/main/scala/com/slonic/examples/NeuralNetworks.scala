package com.slonic.examples

import breeze.linalg.DenseVector
import com.slonic.NeuralNetwork.{ANN, Perceptron}
import com.slonic.datasets.{Digits5k, Digits}
import com.slonic.Metrics.accuracy

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
  val nn = new ANN(name = "hid2", train, labels, nHidden = 1, layerSize = 25)

  nn.train(lambda = 0.8, restoreSnapshot = false)
  val yPred = nn.predict(train)
  println("Accuracy: " + accuracy(labels, DenseVector(yPred:_*)))
}
