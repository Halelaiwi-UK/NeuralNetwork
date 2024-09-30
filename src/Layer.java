import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

public class Layer {
    Neuron[] neurons;
    int layer_size;

    public Layer(int layer_size, int input_size){
        this.layer_size = layer_size;
        this.neurons = new Neuron[this.layer_size];
        for (int i = 0; i < layer_size; i++) {
            this.neurons[i] = new Neuron(input_size);
        }
    }

    public Layer(int layer_size, int input_size, String activationFunction){
        this.layer_size = layer_size;
        this.neurons = new Neuron[this.layer_size];
        for (int i = 0; i < layer_size; i++) {
            this.neurons[i] = new Neuron(input_size, activationFunction);
        }
    }

    // feed the input to each neuron in the layer and store their outputs in a vector
    public RealVector compute_layer(RealVector input_vector){
        double[] output = new double[this.layer_size];
        for (int i = 0; i < this.layer_size; i++) {
            output[i] = this.neurons[i].compute_neuron(input_vector);
        }
        // Convert to a vector before outputting
        return new ArrayRealVector(output);
    }

    public void setWeights(RealVector[] weights){
        for (int i = 0; i < this.layer_size; i++) {
            this.neurons[i].weights = weights[i];
        }
    }

    public void print_weights(){
        for (int i = 0; i < this.layer_size; i++) {
            for (int j = 0; j < this.neurons[i].weights.getDimension(); j++) {
                System.out.println(this.neurons[i].weights.getEntry(j));
            }
        }
    }
}
