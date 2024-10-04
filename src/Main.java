import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Main {
    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork(2, 2);
        network.add_layer(1);
        Array2DRowRealMatrix[] weights = network.getWeights();
        for (Array2DRowRealMatrix weight : weights) {
            System.out.println(weight.toString());
        }
        for (int i =  0; i < network.layers.size(); i++) {
            System.out.println();
            System.out.println("Layer " + (i+1) + ": ");
            network.layers.get(i).print_weights();
        }
    }
}