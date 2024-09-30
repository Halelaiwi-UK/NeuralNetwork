import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.util.*;

public class Neuron {
    RealVector weights;
    double bias;

    // Initialize the neuron with random weights and bias
    public Neuron(int input_size){
        Random random = new Random();

        // Create a random array, then map the values to a random Gaussian distribution
        double[] init_weights = random.doubles(input_size).map(randomValue -> random.nextGaussian()).toArray();
        this.weights = new ArrayRealVector(init_weights);
        this.bias = random.nextGaussian();
    }

    public double compute_neuron(RealVector input_array){
        System.out.println(this.weights.toString());
        if (input_array.getDimension() != this.weights.getDimension()) {
            throw new IllegalArgumentException("Mismatched vector sizes for dot product operation: "
                    + input_array.getDimension() + " != " + this.weights.getDimension());
        }

        return this.weights.dotProduct(input_array) + this.bias;
    }
}
