import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;

public class NeuralNetwork {
    ArrayList<Layer> layers;
    int prev_input_size;
    RealVector output;
    public NeuralNetwork(int first_layer_size, int input_size){
        this.layers = new ArrayList<>();
        this.layers.add(new Layer(first_layer_size, input_size));
        this.prev_input_size = input_size;
    }

    public void add_layer(int layer_size){
        this.layers.add(new Layer(layer_size,this.prev_input_size));
        this.prev_input_size = layer_size;
    }

    public void set_weights(RealVector[][] weights){
        for (int i = 0; i < this.layers.size(); i++) {
            this.layers.get(i).setWeights(weights[i]);
        }
    }

    public RealVector compute_network(double[] input){
        output = new ArrayRealVector(input);
        for (Layer layer: this.layers) {
            output = layer.compute_layer(output);
        }
        return output;
    }
    // Method to print the network structure as a grid with centered neurons
    public void printNetworkGrid(int maxNeuronsInLayer) {
        for (int i = 0; i < this.layers.size(); i++) {
            // Calculate padding spaces needed to center-align the row
            int padding = (maxNeuronsInLayer - this.layers.get(i).layer_size) / 2;

            // Print leading spaces for centering
            for (int j = 0; j < padding; j++) {
                System.out.print("  ");
            }

            // Print "o" for each neuron in the current layer
            for (int j = 0; j < this.layers.get(i).layer_size; j++) {
                System.out.print(" o ");
            }

            System.out.println();
        }
    }
}
