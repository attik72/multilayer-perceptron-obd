package nm;

import iris.DataLoader;

import org.neuroph.core.learning.TrainingElement;
import org.neuroph.core.learning.TrainingSet;
import org.neuroph.nnet.MultiLayerPerceptron;

public class Application {

	public static void log(String message){
		EasyLogger.log(message);
	}

	public static void main(String[] args) {
		log("starting...");

		ThreeLayerPerceptron perceptron = new ThreeLayerPerceptron();		
		perceptron.runOptimalBrainDamage();
		perceptron.testNeuralNetwork(perceptron.getDataLoader().getVersicolorTS());
		log("done");
	}
}
