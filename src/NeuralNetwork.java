import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;

public class NeuralNetwork {
    ArrayList<Layer> layers;
    int prev_input_size;
    RealVector output;
    String lossFunction;
    double loss;

    public NeuralNetwork(int first_layer_size, int input_size){
        this.layers = new ArrayList<>();
        this.layers.add(new Layer(first_layer_size, input_size));
        this.prev_input_size = input_size;
    }

    public NeuralNetwork(int first_layer_size, int input_size, String activationFunction){
        this.layers = new ArrayList<>();
        this.layers.add(new Layer(first_layer_size, input_size, activationFunction));
        this.prev_input_size = input_size;
    }

    public void add_layer(int layer_size){
        this.layers.add(new Layer(layer_size,this.prev_input_size));
        this.prev_input_size = layer_size;
    }
    public void add_layer(int layer_size, String activationFunction){
        this.layers.add(new Layer(layer_size,this.prev_input_size, activationFunction));
        this.prev_input_size = layer_size;
    }

    public void set_weights(Array2DRowRealMatrix[] weights){
        for (int i = 0; i < this.layers.size(); i++) {
            this.layers.get(i).setWeights(weights[i]);
        }
    }
    public Array2DRowRealMatrix[] getWeights(){
        Array2DRowRealMatrix[] weights = new Array2DRowRealMatrix[this.layers.size()];
        for (int i = 0; i < this.layers.size(); i++) {
            weights[i] = this.layers.get(i).getWeights();
        }
        return weights;
    }

    public RealVector compute_network(double[] input){
        // forward feed through the network
        output = new ArrayRealVector(input);
        for (Layer layer: this.layers) {
            output = layer.compute_layer(output);
        }
        return output;
    }
    // Method to print the network structure as a grid with centered neurons
    public void printNetworkGrid(int maxNeuronsInLayer) {
        for (Layer layer : this.layers) {
            // Calculate padding spaces needed to center-align the row
            int padding = (maxNeuronsInLayer - layer.layer_size) / 2;

            // Print leading spaces for centering
            for (int j = 0; j < padding; j++) {
                System.out.print("  ");
            }

            // Print "o" for each neuron in the current layer
            for (int j = 0; j < layer.layer_size; j++) {
                System.out.print(" o ");
            }

            System.out.println();
        }
    }

    private void calculateLoss(double[] expected){
        RealVector expectedVector = new ArrayRealVector(expected);
        if (this.lossFunction.equalsIgnoreCase("mse")){
            this.loss = meanSquaredError(expected);
        }
        else{
            this.loss = meanAbsoluteError(expected);
        }
    }
    public void setLossFunction(String lossFunction) {
        this.lossFunction = lossFunction;
    }


    private double meanSquaredError(double[] expectedOutput){
        RealVector expectedOutputVector = new ArrayRealVector(expectedOutput);
        RealVector diff = expectedOutputVector.subtract(this.output);
        return (diff.getNorm() * diff.getNorm()) / expectedOutputVector.getDimension();
    }
    private double meanAbsoluteError(double[] expectedOutput){
        RealVector expectedOutputVector = new ArrayRealVector(expectedOutput);
        RealVector diff = expectedOutputVector.subtract(this.output);
        return diff.getL1Norm() / expectedOutputVector.getDimension();
    }
}
