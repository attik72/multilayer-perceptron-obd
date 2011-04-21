package iris;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.neuroph.core.learning.TrainingElement;
import org.neuroph.core.learning.TrainingSet;
import org.neuroph.util.TrainingSetImport;

import java.util.Iterator;

public class Virginica {
	
	private TrainingSet trainingSet = new TrainingSet();
	static String name = "iris_virginica.data";
	private TrainingSet learningTS = new TrainingSet();
	private TrainingSet testTS = new TrainingSet();

	public Virginica(int numLearningEl) {
		 try {
             trainingSet = TrainingSetImport.importFromFile(Virginica.name, 4, 1, ",");
             
             for (Iterator<TrainingElement> it = trainingSet.iterator (); it.hasNext(); ) {
     			if (this.learningTS.getRecordCount() < numLearningEl) {
     				this.learningTS.addElement(it.next());
     			} else {
     				this.testTS.addElement(it.next());
     			}
     		 }
		 } catch (NumberFormatException e) {
             // TODO Auto-generated catch block
             e.printStackTrace();
		 } catch (FileNotFoundException e) {
             // TODO Auto-generated catch block
             e.printStackTrace();
		 } catch (IOException e) {
             // TODO Auto-generated catch block
             e.printStackTrace();
		 }
		
	}

	public TrainingSet getTrainingSet() {
		return trainingSet;
	}

	public void setTrainingSet(TrainingSet trainingSet) {
		this.trainingSet = trainingSet;
	}

	public TrainingSet getLearningTS() {
		return learningTS;
	}

	public void setLearningTS(TrainingSet learningTS) {
		this.learningTS = learningTS;
	}

	public TrainingSet getTestTS() {
		return testTS;
	}

	public void setTestTS(TrainingSet testTS) {
		this.testTS = testTS;
	}
	
	
}
