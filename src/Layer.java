import ActivationFunctions.ActivationFunction;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
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

    public Layer(int layer_size, int input_size, ActivationFunction activationFunction){
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

    public void setWeights(Array2DRowRealMatrix weights){
        for (int i = 0; i < this.layer_size; i++) {
            this.neurons[i].setWeights(weights.getRow(i));
        }
    }

    public Array2DRowRealMatrix getWeights(){
        Array2DRowRealMatrix weights = new Array2DRowRealMatrix(this.layer_size, this.neurons[0].getWeights().getDimension());
        for (int i = 0; i < this.layer_size; i++) {
            weights.setRow(i, this.neurons[i].getWeights().toArray());
        }
        return weights;
    }
    public void print_weights(){
        for (int i = 0; i < this.layer_size; i++) {
            System.out.print("   ");
            System.out.print("Neuron " + (i+1) + " {");
            for (int j = 0; j < this.neurons[i].getWeights().getDimension(); j++) {
                System.out.print(this.neurons[i].getWeights().getEntry(j) + ", ");
            }
            System.out.println("} bias {" + this.neurons[i].getBias() + "}");
        }
    }
}
