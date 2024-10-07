import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

/**
 * Class to handle all mathematical loss and activation functions
 */

public class CustomFunctions {

    public static double activation_function(double output, String activationFunction){
        if (activationFunction.equalsIgnoreCase("relu")){
            return ReLu(output);
        } else if (activationFunction.equalsIgnoreCase("sigmoid")) {
            return Sigmoid(output);
        } else if (activationFunction.equalsIgnoreCase("leaky_relu")) {
            return LeakyRelu(output);
        } else if (activationFunction.equalsIgnoreCase("tanh")) {
            return Tanh(output);
        }
        return output;
    }

    public static double activation_function(double output, String activationFunction, double alpha){
        if (activationFunction.equalsIgnoreCase("relu")){
            return ReLu(output);
        } else if (activationFunction.equalsIgnoreCase("sigmoid")) {
            return Sigmoid(output);
        } else if (activationFunction.equalsIgnoreCase("leaky_relu")) {
            return LeakyRelu(output, alpha);
        } else if (activationFunction.equalsIgnoreCase("tanh")) {
            return Tanh(output);
        }
        return output;
    }

    public static double loss_function(RealVector predicted, RealVector actual, String lossFunction){
        if (lossFunction.equalsIgnoreCase("mse")){
            return meanSquaredError(predicted, actual);
        }
        else if (lossFunction.equalsIgnoreCase("mae")){
            return meanAbsoluteError(predicted, actual);
        }
        // Fallback function
        return meanSquaredError(predicted, actual);
    }

    private static double ReLu(double output){
        if (output < 0){
            return 0;
        }
        return output;
    }

    private static double LeakyRelu(double output){
        if (output < 0){
            return 0.1 * output;
        }
        return output;
    }

    private static double LeakyRelu(double output, double alpha){
        if (output < 0){
            return alpha * output;
        }
        return output;
    }

    private static double Sigmoid(double output){
        return 1/(1 + Math.exp(-output));
    }

    private static double Tanh(double output){
        return Math.tanh(output);
    }

    private static double meanSquaredError(RealVector predictedOutput, RealVector expectedOutput){
        RealVector diff = expectedOutput.subtract(predictedOutput);
        return (diff.getNorm() * diff.getNorm()) / expectedOutput.getDimension();
    }
    private static double meanAbsoluteError(RealVector predictedOutput, RealVector expectedOutput){
        RealVector diff = expectedOutput.subtract(predictedOutput);
        return diff.getL1Norm() / expectedOutput.getDimension();
    }
}
