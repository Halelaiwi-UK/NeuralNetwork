package LossFunctions;

import org.apache.commons.math3.linear.RealVector;

public class MeanSquaredError implements LossFunction {
    public MeanSquaredError() {}

    @Override
    public double computeLoss(RealVector predicted, RealVector actual) {
        RealVector diff = actual.subtract(predicted);
        return (diff.getNorm() * diff.getNorm()) / actual.getDimension();
    }

    @Override
    public RealVector derivative(RealVector predicted, RealVector actual) {
        return null;
    }
}
