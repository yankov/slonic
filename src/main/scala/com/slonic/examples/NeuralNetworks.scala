package com.slonic.examples

import com.slonic.NeuralNetwork.Perceptron
import com.slonic.datasets.Digits

object NeuralNetworks extends App {
  println("Perceptron example")
  val predLabel = 4
  val (labels, train) = Digits.load()
  val y = labels.map(l => if(l == predLabel) 1 else 0)

  val perceptron = new Perceptron()
  perceptron.fit(train, y)
  val yPred = perceptron.predict(train)
  println(s"Weights: ${perceptron.weights}")

}
