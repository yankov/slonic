package com.slonic.examples

import com.slonic.NeuralNetwork.Perceptron
import com.slonic.datasets.Digits
import com.slonic.Metrics.accuracy

object Perceptron extends App {
   println("Perceptron example")

   val (labels, train) = Digits.load()

   // Predict class 9 (currently it can only do binary classification)
   val predLabel = 9

   val y = labels.map(l => if(l == predLabel) 1 else 0)
   val perceptron = new Perceptron()
   perceptron.fit(train, y)
   val yPred = perceptron.predict(train)
   println(s"Accuracy: ${accuracy(y, yPred)}")
}
