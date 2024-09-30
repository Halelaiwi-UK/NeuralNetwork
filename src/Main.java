import org.apache.commons.math3.linear.RealVector;

public class Main {
    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork(2, 2);
        network.add_layer(1);
        RealVector result = network.compute_network(new double[]{5, 3});
        System.out.println(result.toString());
        network.printNetworkGrid(2);
    }
}