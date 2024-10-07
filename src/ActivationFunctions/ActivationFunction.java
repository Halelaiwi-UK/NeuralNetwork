package ActivationFunctions;

public interface ActivationFunction {
    double apply(double input);
    double derivative(double input);  // Derivative for backpropagation
}
