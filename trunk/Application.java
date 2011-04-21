package nm;

import java.sql.Date;
import java.text.SimpleDateFormat;
import java.util.Calendar;

import iris.Setosa;
import iris.Versicolor;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;


public class Application {
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		MultiLayerPerceptron neuralNetwork = new MultiLayerPerceptron(TransferFunctionType.LINEAR ,4,1,1);// Perceptron(4, 1, TransferFunctionType.STEP); 
		// create training set 
		
		Setosa vc = new Setosa(30);
		 
		// learn the training set 
		System.out.println("vreme startovanja: " + now());
		neuralNetwork.learnInSameThread(vc.getLearningTS());
		System.out.println("zavrsio ucenje: " + now());
		neuralNetwork.save("izuceno.nnet");
		System.out.println("sacuvao u fajl: " + now());
		
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
	
	public static String now() {
	    Calendar cal = Calendar.getInstance();
	    SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
	    return sdf.format(cal.getTime());

	  }

}
