package ActivationFunctions;

public class Sigmoid implements ActivationFunction {
    @Override
    public double apply(double input) {
        return 1 / (1 + Math.exp(-input));
    }

    @Override
    public double derivative(double input) {
        return input * (1 - input);  // Sigmoid derivative
    }
}
