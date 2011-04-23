package nm;

import iris.DataLoader;

import java.util.HashMap;
import java.util.List;
import java.util.Vector;

import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
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
		this.base = NeuralNetworkFactory.createMLPerceptron("4 10 1", TransferFunctionType.SIGMOID, BackPropagation.class, true, true);
	}
	
	// this is the method that does all the work
	public void runOptimalBrainDamage(){
		if(isOptimizationNeeded()){
			rinseAndRepeat();
		}else{
			// rollback
			this.base = (MultiLayerPerceptron) MultiLayerPerceptron.load(BASE_STORAGE_PATH);
			// TODO this is the optimal base, now do whatever you want with it
		}
	}
	
	public boolean isOptimizationNeeded() {
		return isErrorAcceptable();
	}
	
	// has the limit been reached?
	public boolean isErrorAcceptable(){
		return this.error < ERROR_THRESHOLD;
	}

	// single optimal brain damage iteration
	public void rinseAndRepeat(){
		// do we need to delete file contents prior to writing?
		base.save(BASE_STORAGE_PATH);
		Connection conn = getTheLeastImportantConnection();
		removeConnection(conn);
	}
	
	// connections are links between neurons. This will return
	// the one which once removed changes the global (neural network) error the least
	public Connection getTheLeastImportantConnection(){
		HashMap importanceHash = getConnectionsAndTheirImportance();
		return getTheLeastImportantConnectionFrom(importanceHash);
	}
	
	// hidden forward connections + their formula based importance
	public HashMap getConnectionsAndTheirImportance(){
		Vector connections = getHiddenForwardConnections();
		HashMap importanceHash = new HashMap();
		Connection currentConnection = null;
		for(int i=0; i < connections.size(); i++) {
			currentConnection = (Connection) connections.get(i);
			importanceHash.put(currentConnection, getImportanceFor(currentConnection));
		}
		return importanceHash;
		
	}
	
	public double getImportanceFor(Connection connection) {
		// TODO formula goes here
		return 0;
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
	public Connection getTheLeastImportantConnectionFrom(HashMap importanceHash){
		Object[] connections = importanceHash.values().toArray();
		Object[] importances = importanceHash.values().toArray();
		double lowestImportance = minimumFrom(importances);
		for(int i=0; i<connections.length; i++){
			Object currentConnection = connections[i];
			if(importanceHash.get(currentConnection).equals(lowestImportance)){
				return (Connection) currentConnection;
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
		conn.getToNeuron().removeInputConnectionFrom(conn.getFromNeuron());
	}
	
	public Layer getHiddenLayer(){
		return base.getLayerAt(1);
	}
	
	// example: learn("setosa", 30)
	// total number of samples - 30 will be preserved for testing
	public void learn(String trainingSetName, int howManyToLearn){
		DataLoader dataLoader = new DataLoader(trainingSetName, howManyToLearn);
		base.learnInSameThread(dataLoader.getLearningTS());
	}
	
	// save the base to the custom file
	public void save(String filepath) {
		base.save(filepath);
	}
}
