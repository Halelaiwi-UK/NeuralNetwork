import ActivationFunctions.*;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.util.Random;

public class Neuron {
    private RealVector weights;
    private ActivationFunction activationFunction;
    private double bias;
    // Learning rate
    double alpha = 0.1;

    // Initialize the neuron with random weights and bias
    public Neuron(int input_size){
        Random random = new Random();

        // Create a random array, then map the values to a random Gaussian distribution
        double[] init_weights = random.doubles(input_size).map(_ -> random.nextGaussian()).toArray();
        this.weights = new ArrayRealVector(init_weights);
        this.bias = random.nextGaussian();

        this.activationFunction = new Sigmoid(); // default to sigmoid
    }

    public Neuron(int input_size, ActivationFunction activation_function){
        Random random = new Random();
        // Create a random array, then map the values to a random Gaussian distribution
        double[] init_weights = random.doubles(input_size).map(_ -> random.nextGaussian()).toArray();
        this.weights = new ArrayRealVector(init_weights);
        this.bias = random.nextGaussian();

        this.activationFunction = activation_function;
    }

    public double compute_neuron(RealVector input_array){
        if (input_array.getDimension() != this.weights.getDimension()) {
            throw new IllegalArgumentException("Mismatched vector sizes for dot product operation: "
                    + input_array.getDimension() + " != " + this.weights.getDimension());
        }

        return this.activationFunction.apply(this.weights.dotProduct(input_array) + this.bias);
    }

    public void setWeights(double[] new_weights) {
        this.weights = new ArrayRealVector(new_weights);
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

}
