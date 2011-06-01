package nm;

import org.neuroph.core.learning.TrainingElement;
import org.neuroph.core.learning.TrainingSet;

public class Application {
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		log("starting...");

		ThreeLayerPerceptron perceptron = new ThreeLayerPerceptron();
		perceptron.runOptimalBrainDamage();
		
		TrainingSet ts = new TrainingSet();
		// ts.addElement(new TrainingElement("1 1 1 1"));
		// perceptron.test(new TrainingSet(), expected)
	    // log("learning finished");
		// perceptron.save("izuceno.nnet");
		
		log("done");
		
		/*
		// load the saved network 
		NeuralNetwork neuralNetwork1 = NeuralNetwork.load("or_perceptron.nnet"); */
		// set network input 
		/*neuralNetwork.setInput(6.5,2.8,4.6,1.5); 
		// calculate network 
		neuralNetwork.calculate(); 
		// get network output 
		double[] networkOutput = neuralNetwork.getOutput(); 
		
		System.out.println(networkOutput[0]);*/
		
	}
	
	public static void log(String message){
		EasyLogger.log(message);
	}
}
