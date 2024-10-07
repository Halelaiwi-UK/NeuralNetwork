import LossFunctions.*;
import optimizers.Optimizer;

public class Trainer {
    private NeuralNetwork neuralNetwork;
    private LossFunction lossFunction;
    private Optimizer optimizer;
    private double learningRate;
    private int batchSize;
    private int epochs;

    public Trainer(NeuralNetwork neuralNetwork, LossFunction lossFunction, Optimizer optimizer, double learningRate, int batchSize, int epochs) {
        this.neuralNetwork = neuralNetwork;
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.epochs = epochs;
    }

}

