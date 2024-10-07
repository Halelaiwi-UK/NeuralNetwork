package ActivationFunctions;

public class ReLU implements ActivationFunction {
    @Override
    public double apply(double input) {
        return Math.max(0, input);
    }

    @Override
    public double derivative(double input) {
        return input > 0 ? 1 : 0;
    }
}