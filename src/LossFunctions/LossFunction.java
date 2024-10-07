package LossFunctions;

import org.apache.commons.math3.linear.RealVector;

public interface LossFunction{
    double computeLoss(RealVector predicted, RealVector actual);
    RealVector derivative(RealVector predicted, RealVector actual);
}
