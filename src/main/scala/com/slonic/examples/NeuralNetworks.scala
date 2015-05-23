package com.slonic.examples

import breeze.linalg.DenseVector
import com.slonic.NeuralNetwork.ANN
import com.slonic.datasets.{Digits5k, Digits}
import com.slonic.Metrics.accuracy

object NeuralNetworks extends App {
  val (labels, train) = Digits5k.load()

  println("Multi-layer neural net example")
  val nn = new ANN(name = "hid2", train, labels, nHidden = 1, layerSize = 25)

  nn.train(alpha = 0.8, restoreSnapshot = false)
  val yPred = nn.predict(train)
  println("Accuracy: " + accuracy(labels, DenseVector(yPred:_*)))
}
