package nm;

import iris.DataLoader;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.learning.SupervisedTrainingElement;
import org.neuroph.core.learning.TrainingElement;
import org.neuroph.core.learning.TrainingSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.TransferFunctionType;

public class ThreeLayerPerceptron {
	
	// the maximum error allowed
	private static final double ERROR_THRESHOLD = 0.01;
	
	// when optimizing, we are saving the base so we can easily
	// rollback when the error gets unsatisfactory
	private static final String BASE_STORAGE_PATH = "base_backup";
	private static final int MAX_ITERATIONS = 20000;
	
	// ThreeLayerPerceptron is a wrapper around this class provided by
	// the neuroph framework
	private MultiLayerPerceptron base;
	
	private DataLoader dataLoader;
	
	public DataLoader getDataLoader() {
		return dataLoader;
	}

	public void setDataLoader(DataLoader dataLoader) {
		this.dataLoader = dataLoader;
	}

	// NeuralNetworkFactory allows for the most flexibility
	// default formation 4-10-1
	public ThreeLayerPerceptron() {
		this.base = NeuralNetworkFactory.createMLPerceptron("4 10 3", TransferFunctionType.SIGMOID, BackPropagation.class, true, true);
		((BackPropagation)this.base.getLearningRule()).setMaxIterations(MAX_ITERATIONS);
	}
	
	// this is the method that does all the work
	public void runOptimalBrainDamage(){
		System.out.println("Broj neurona pre optimizacije: " + this.getHiddenNeuronCount());
		
		while (isOptimizationNeeded() && rinseAndRepeat()){
			System.out.println("iterating obd " + getHiddenNeuronCount());
		}
		
/*		for(int i = 0; i < 5; i++){
			rinseAndRepeat();
		}*/
		
		// rollback
		System.out.println("rollback");
		this.base = (MultiLayerPerceptron) MultiLayerPerceptron.load(BASE_STORAGE_PATH);
		System.out.println("Broj neurona posle optimizacije: " + this.getHiddenNeuronCount());
	}
	
	public boolean isOptimizationNeeded() {
		return (getHiddenNeuronCount() > 0) && isErrorAcceptable();
	}
	
	// has the limit been reached?
	public boolean isErrorAcceptable(){
		double error = ((BackPropagation)this.base.getLearningRule()).getTotalNetworkError();
		System.out.println(error);
		return error < ERROR_THRESHOLD;
	}

	// single optimal brain damage iteration
	public boolean rinseAndRepeat(){
		// do we need to delete file contents prior to writing?
		// learning
				
		System.out.println("rinseAndRepeat");
		dataLoader = learn(30);	
		System.out.println("izvrsio learn");
		if(isErrorAcceptable()) {			
			Connection conn = getTheLeastImportantConnection();
			if (conn == null) return false;
			save(BASE_STORAGE_PATH);
			removeConnection(conn);
		}		

		return true;
	}
	
	public double test(TrainingSet ts, double expected) {
		double totalError = 0;
		for(TrainingElement trainingElement : ts.trainingElements()) {
			base.setInput(trainingElement.getInput()[0], trainingElement.getInput()[1], trainingElement.getInput()[2], trainingElement.getInput()[3]);
			base.calculate();
			base.notifyChange();
			/*double[] networkOutput = base.getOutput();
			System.out.println(networkOutput[0] + " : " + expected);
			totalError += expected -  Math.abs(networkOutput[0]);*/
		}
		return ((BackPropagation)base.getLearningRule()).getTotalNetworkError();
		// return totalError/ts.trainingElements().size();
	}
	
	// connections are links between neurons. This will return
	// the one which once removed changes the global (neural network) error the least
	public Connection getTheLeastImportantConnection(){
		Map saliencyHash = getConnectionsAndTheirSaliency();
		return getTheLeastImportantConnectionFrom(saliencyHash);
	}
	
	// hidden forward connections + their formula based importance
	public Map getConnectionsAndTheirSaliency(){
		Vector connections = getHiddenForwardConnections();
		Map saliencyHash = new HashMap();
		Connection currentConnection = null;
		for(int i=0; i < connections.size(); i++) {
			currentConnection = (Connection) connections.get(i);
			saliencyHash.put(currentConnection, getSaliencyFor(currentConnection));
		}
		return saliencyHash;
		
	}
	
	public double getSaliencyFor(Connection connection) {
		double saliency =  0.5 * hessianMatrix(connection) * (connection.getWeight().getValue() * connection.getWeight().getValue());
		return saliency;
	}
	
	public double hessianMatrix(Connection connection) {
		double hessian = 0;
		Neuron hiddenNeuron = connection.getFromNeuron();
		List<Connection> connections = hiddenNeuron.getInputConnections();
		for(int i = 0; i < connections.size(); i++) {
			double a = connections.get(i).getWeightedInput();
			hessian += 2 * sigmoidDerivative(a) * sigmoidDerivative(a) * connections.get(i).getInput() * connections.get(i).getInput();
		}
		return hessian;
	}
	
	public double sigmoid(double x){
		return 1.0 / (1.0 + Math.exp(-x));
	}
	
	public double sigmoidDerivative(double x) {
		return sigmoid(x) * (1 - sigmoid(x));
	}
	
	// since there is only one hidden layer all hidden forward connections
	// will be between middle and output layer
	public Vector getHiddenForwardConnections(){
		Vector connections = new Vector();
		Neuron currentHiddenNeuron = null;
		for(int i=0; i<getHiddenNeuronCount(); i++) {
			currentHiddenNeuron = getHiddenNeuronAt(i);
			List<Connection> connectionsForThisNeuron = currentHiddenNeuron.getOutConnections();
			for(int j=0; j<connectionsForThisNeuron.size(); j++) {
				connections.add(connectionsForThisNeuron.get(j));
			}
		}
		return connections;
	}
	
	// get number of neurons in the hidden layer
	public int getHiddenNeuronCount(){
		return getHiddenLayer().getNeuronsCount();
	}
	
	// get a specific neuron in the hidden layer by its index
	public Neuron getHiddenNeuronAt(int idx){
		return getHiddenLayer().getNeuronAt(idx);
	}
	
	// importanceHash is a map in this format:
	// connection -> value
	// the least valuable connection will be returned
	public Connection getTheLeastImportantConnectionFrom(Map importanceHash){
		if (importanceHash.size() == 0) return null;
		Iterator it = importanceHash.entrySet().iterator();
		Object[] importances = importanceHash.values().toArray();
		double lowestImportance = minimumFrom(importances);
	    while (it.hasNext()) {
	        Map.Entry pairs = (Map.Entry)it.next();
	        // System.out.println(pairs.getKey() + " = " + pairs.getValue());
	        if(pairs.getValue().equals(lowestImportance)) {
	        	return (Connection)pairs.getKey();
	        }
	    }
	    return null;
	}
	
	// array minimum
	public double minimumFrom(Object[] importances){
		double currentMinimum = (Double) importances[0];
		double currentValue;
		for(int i=1; i<importances.length; i++) {
			currentValue = (Double) importances[0];
			if(currentValue < currentMinimum) {
				currentMinimum = currentValue;
			}
		}
		return currentMinimum;
	}
	
	public void removeConnection(Connection conn){
		Neuron neuron = conn.getFromNeuron();
		conn.getToNeuron().removeInputConnectionFrom(neuron);
		neuron.getOutConnections().remove(conn);
		for(int i = 0; i < this.getHiddenNeuronCount(); i++) {
			Neuron temp = this.getHiddenNeuronAt(i);
			if (temp.getOutConnections().size() == 0) {
				this.getHiddenLayer().removeNeuron(temp);
			}
		}
	}
	
	public Layer getHiddenLayer(){
		return base.getLayerAt(1);
	}
	
	// example: learn("setosa", 30)
	// total number of samples - 30 will be preserved for testing
	public DataLoader learn(int howManyToLearn){
		DataLoader dataLoader = new DataLoader(howManyToLearn);
		base.learnInSameThread(dataLoader.getLearningTS());
		return dataLoader;
	}
	
	// save the base to the custom file
	public void save(String filepath) {
		System.out.println("save " + getHiddenNeuronCount());
		base.save(filepath);
	}
	
	public static void main(String[] args) {
		
		TrainingSet trainingSet = (new DataLoader(40)).getLearningTS();
		
		// create multi layer perceptron
		// MultiLayerPerceptron myMlPerceptron = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 4, 2, 3);
		
		ThreeLayerPerceptron perceptron = new ThreeLayerPerceptron();
		MultiLayerPerceptron myMlPerceptron = perceptron.base;
		
		perceptron.runOptimalBrainDamage();
		
		testNeuralNetwork(myMlPerceptron, perceptron.dataLoader.getVersicolorTS());
		
		/*System.out.println(((BackPropagation)myMlPerceptron.getLearningRule()).getTotalNetworkError());

		// test perceptron
		System.out.println("Testing trained neural network");
		testNeuralNetwork(myMlPerceptron, trainingSet);

		// save trained neural network
		myMlPerceptron.save("myMlPerceptron.nnet");

		// load saved neural network
		NeuralNetwork loadedMlPerceptron = NeuralNetwork.load("myMlPerceptron.nnet");

		// test loaded neural network
		System.out.println("Testing loaded neural network");
		testNeuralNetwork(loadedMlPerceptron, trainingSet);
		System.out.println(((BackPropagation)myMlPerceptron.getLearningRule()).getTotalNetworkError());
*/
	}
	
	public static void testNeuralNetwork(NeuralNetwork nnet, TrainingSet tset) {

		for(TrainingElement trainingElement : tset.trainingElements()) {

		nnet.setInput(trainingElement.getInput());
		nnet.calculate();
		nnet.notifyChange();
		double[] networkOutput = nnet.getOutput();
		System.out.print("Input: " + trainingElement.getInput()[0] + trainingElement.getInput()[1] + trainingElement.getInput()[2] + trainingElement.getInput()[3]);
		System.out.println(" Output: " + getIrisType(networkOutput));

		}

	}
	
	public static String getIrisType(double[] networkOutput) {
		// maximum
		double max = (networkOutput[0] > networkOutput[1]) ? networkOutput[0] : networkOutput[1];
		max =  (networkOutput[2] > max) ? networkOutput[2] : max;
		
		if(max == networkOutput[0])
			return "Setosa" +  networkOutput[0] +  networkOutput[1] +  networkOutput[2];
		if(max == networkOutput[1])
			return "Versicolor" + networkOutput[0] +  networkOutput[1] +  networkOutput[2];
		if(max == networkOutput[2])
			return "Virginica" + networkOutput[0] +  networkOutput[1] +  networkOutput[2];
		
		return "NE ZNAM!!!";
		
	}
}
