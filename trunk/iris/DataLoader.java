package iris;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.neuroph.core.learning.TrainingElement;
import org.neuroph.core.learning.TrainingSet;
import org.neuroph.util.TrainingSetImport;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class DataLoader {

	static String loadpath;
	private TrainingSet trainingSet = new TrainingSet();
	private TrainingSet learningTS = new TrainingSet();
	private TrainingSet setosaTS = new TrainingSet();
	private TrainingSet versicolorTS = new TrainingSet();
	private TrainingSet virginicaTS = new TrainingSet();

	public DataLoader(int numLearningEl) {
		 try {
			 
             TrainingSet setosaSet = TrainingSetImport.importFromFile("iris_setosa.data", 4, 1, ",");
             TrainingSet versicolorSet = TrainingSetImport.importFromFile("iris_versicolor.data", 4, 1, ",");
             TrainingSet virginicaSet = TrainingSetImport.importFromFile("iris_virginica.data", 4, 1, ",");
            	 
             for(int i = 0; i < setosaSet.trainingElements().size(); i++) {
            	 if(i < numLearningEl)
            		 this.learningTS.addElement(setosaSet.trainingElements().get(i));
            	 else
            		 this.setosaTS.addElement(setosaSet.trainingElements().get(i));
             }
             
             for(int i = 0; i < versicolorSet.trainingElements().size(); i++) {
            	 if(i < numLearningEl)
            		 this.learningTS.addElement(versicolorSet.trainingElements().get(i));
            	 else
            		 this.versicolorTS.addElement(versicolorSet.trainingElements().get(i));
             }
             
             for(int i = 0; i < virginicaSet.trainingElements().size(); i++) {
            	 if(i < numLearningEl)
            		 this.learningTS.addElement(virginicaSet.trainingElements().get(i));
            	 else
            		 this.virginicaTS.addElement(versicolorSet.trainingElements().get(i));
                       
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

	public TrainingSet getSetosaTS() {
		return setosaTS;
	}

	public void setSetosaTS(TrainingSet setosaTS) {
		this.setosaTS = setosaTS;
	}

	public TrainingSet getVersicolorTS() {
		return versicolorTS;
	}

	public void setVersicolorTS(TrainingSet versicolorTS) {
		this.versicolorTS = versicolorTS;
	}

	public TrainingSet getVirginicaTS() {
		return virginicaTS;
	}

	public void setVirginicaTS(TrainingSet virginicaTS) {
		this.virginicaTS = virginicaTS;
	}

	
	
	
}
