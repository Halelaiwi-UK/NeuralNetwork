package LossFunctions;

import org.apache.commons.math3.linear.RealVector;

public class CrossEntropyLoss implements LossFunction {

    // Small epsilon to avoid log(0) issues
    private static final double EPSILON = 1e-9;

    @Override
    public double computeLoss(RealVector predicted, RealVector actual) {
        // Apply epsilon for numerical stability
        RealVector adjustedPredicted = predicted.mapAdd(EPSILON);

        RealVector logPredicted = adjustedPredicted.mapToSelf(Math::log);
        // Cross entropy loss is the negative dot product between actual and logPredicted
        return -actual.dotProduct(logPredicted);
    }

    @Override
    public RealVector derivative(RealVector predicted, RealVector actual) {
        // Derivative of cross-entropy loss with respect to predicted values
        RealVector adjustedPredicted = predicted.mapAdd(EPSILON);

        // Element-wise division: -(actual / predicted)
        return actual.ebeDivide(adjustedPredicted).mapMultiplyToSelf(-1);
    }
}