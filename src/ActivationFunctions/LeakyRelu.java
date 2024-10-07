package ActivationFunctions;

public class LeakyRelu implements ActivationFunction {
    double alpha = 0.1;

    public LeakyRelu(){
        // Default to 0.1
    }

    public LeakyRelu(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public double apply(double output){
        if (output < 0){
            return 0.1 * output;
        }
        return output;
    }

    @Override
    public double derivative(double input) {
        if (input < 0){
            return this.alpha * input;
        }
        return input;
    }
}
