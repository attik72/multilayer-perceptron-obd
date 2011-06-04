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
		
		log("done");
	}
	
	public static void log(String message){
		EasyLogger.log(message);
	}
}
