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
import org.neuroph.core.Neuron;
import org.neuroph.core.learning.TrainingElement;
import org.neuroph.core.learning.TrainingSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.TransferFunctionType;

public class ThreeLayerPerceptron {
	
	// the maximum error allowed
	private static final double ERROR_THRESHOLD = 0.002;
	
	// when optimizing, we are saving the base so we can easily
	// rollback when the error gets unsatisfactory
	private static final String BASE_STORAGE_PATH = "base_backup";
	
	// ThreeLayerPerceptron is a wrapper around this class provided by
	// the neuroph framework
	private MultiLayerPerceptron base;
	
	// this value should keep increasing and at some point it will
	// get bigger than the threshold. that is a flag that tells us
	// when to stop
	private double error = 0;
	
	// NeuralNetworkFactory allows for the most flexibility
	// default formation 4-10-1
	public ThreeLayerPerceptron() {
		this.base = NeuralNetworkFactory.createMLPerceptron("4 4 1", TransferFunctionType.SIGMOID, BackPropagation.class, true, true);
	}
	
	// this is the method that does all the work
	public void runOptimalBrainDamage(){
		System.out.println("Broj neurona pre optimizacije: " + this.getHiddenNeuronCount());
		
		while (isOptimizationNeeded()){
			System.out.println("iterating");
			rinseAndRepeat();
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
		return isErrorAcceptable();
	}
	
	// has the limit been reached?
	public boolean isErrorAcceptable(){
		System.out.println("error " + error);
		return Math.abs(this.error) < ERROR_THRESHOLD;
	}

	// single optimal brain damage iteration
	public void rinseAndRepeat(){
		// do we need to delete file contents prior to writing?
		// learning
		
		DataLoader dataLoader = null;
		
		for(int i = 0; i<10000; i++){
			dataLoader = learn(44);	
		}
		
		// test
		double setosaError = test(dataLoader.getSetosaTS(), 1);
		double versicolorError = test(dataLoader.getVersicolorTS(), 0.5);
		double virginicaError = test(dataLoader.getVirginicaTS(), 0);
		this.error = 0.333 * (setosaError + versicolorError + virginicaError);
		
		base.save(BASE_STORAGE_PATH);
		
		if(isErrorAcceptable()) {			
			Connection conn = getTheLeastImportantConnection();
			removeConnection(conn);
		}
	}
	
	public double test(TrainingSet ts, double expected) {
		double totalError = 0;
		for(TrainingElement trainingElement : ts.trainingElements()) {

			base.setInput(trainingElement.getInput());
			base.calculate();
			double[] networkOutput = base.getOutput();
			totalError += Math.abs(networkOutput[0] - expected);
			}
		return totalError/ts.trainingElements().size();
	}
	
	// connections are links between neurons. This will return
	// the one which once removed changes the global (neural network) error the least
	public Connection getTheLeastImportantConnection(){
		Map importanceHash = getConnectionsAndTheirImportance();
		return getTheLeastImportantConnectionFrom(importanceHash);
	}
	
	// hidden forward connections + their formula based importance
	public Map getConnectionsAndTheirImportance(){
		Vector connections = getHiddenForwardConnections();
		Map importanceHash = new HashMap();
		Connection currentConnection = null;
		for(int i=0; i < connections.size(); i++) {
			currentConnection = (Connection) connections.get(i);
			importanceHash.put(currentConnection, getImportanceFor(currentConnection));
		}
		return importanceHash;
		
	}
	
	public double getImportanceFor(Connection connection) {
		double importance =  0.5 * hessianMatrix(connection) * (connection.getWeight().getValue() * connection.getWeight().getValue());
		return importance;
	}
	
	public double hessianMatrix(Connection connection) {
		double sum = 0;
		List<Connection> connections = connection.getFromNeuron().getInputConnections();
		for(int i = 0; i < connections.size(); i++) {
			sum += connections.get(i).getWeightedInput();
		}
		return 2 * sigmoidDerivative(sum) * sigmoidDerivative(sum);
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
		base.save(filepath);
	}
}
