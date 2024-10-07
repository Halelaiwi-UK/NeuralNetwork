package ActivationFunctions;

public class Tanh implements ActivationFunction {

    @Override
    public double apply(double input) {
        return Math.tanh(input);
    }

    @Override
    public double derivative(double input) {
        return 1-Math.pow(Math.tanh(input), 2);
    }
}
