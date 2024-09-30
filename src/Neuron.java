import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.util.*;

public class Neuron {
    RealVector weights;
    String activationFunction = "Relu";
    double bias;
    double LeakyRelu_alpha = 0.1;

    // Initialize the neuron with random weights and bias
    public Neuron(int input_size){
        Random random = new Random();

        // Create a random array, then map the values to a random Gaussian distribution
        double[] init_weights = random.doubles(input_size).map(_ -> random.nextGaussian()).toArray();
        this.weights = new ArrayRealVector(init_weights);
        this.bias = random.nextGaussian();
    }

    public Neuron(int input_size, String activation_function){
        Random random = new Random();
        this.activationFunction = activation_function;
        // Create a random array, then map the values to a random Gaussian distribution
        double[] init_weights = random.doubles(input_size).map(_ -> random.nextGaussian()).toArray();
        this.weights = new ArrayRealVector(init_weights);
        this.bias = random.nextGaussian();
    }

    public double compute_neuron(RealVector input_array){
        if (input_array.getDimension() != this.weights.getDimension()) {
            throw new IllegalArgumentException("Mismatched vector sizes for dot product operation: "
                    + input_array.getDimension() + " != " + this.weights.getDimension());
        }

        return activation_function(this.weights.dotProduct(input_array) + this.bias);
    }

    public void setWeights(RealVector weights) {
        this.weights = weights;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public RealVector getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }

    private double activation_function(double output){
        if (this.activationFunction.equalsIgnoreCase("relu")){
            return ReLu(output);
        } else if (this.activationFunction.equalsIgnoreCase("sigmoid")) {
            return Sigmoid(output);
        } else if (this.activationFunction.equalsIgnoreCase("leaky_relu")) {
            return LeakyRelu(output);
        } else if (this.activationFunction.equalsIgnoreCase("tanh")) {
            return Tanh(output);
        }
        return output;
    }
    private double ReLu(double output){
        if (output < 0){
            return 0;
        }
        return output;
    }

    private double LeakyRelu(double output){
        if (output < 0){
            return this.LeakyRelu_alpha * output;
        }
        return output;
    }

    private double Sigmoid(double output){
        return 1/(1 + Math.exp(-output));
    }

    private double Tanh(double output){
        return Math.tanh(output);
    }
}
