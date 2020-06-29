package network

import (
	"errors"
	"fmt"
	"math"
	"neuraldeep/activation"
	"neuraldeep/utils"
	"neuraldeep/utils/matrix"

	"gonum.org/v1/gonum/mat"
)

// A module to implement the stochastic gradient descent learning algorithm for a feedforward neural network.
// Gradients are calculated using backpropagation.
// Note that I tried to stay as close as possible to Michael Nielsen's Python module.

//--- TYPES

// Network defines the neural network structure by instantiating it with an array of sizes,
// ie. the number of neurons per layer
type Network struct {
	Sizes     []int
	numLayers int
	weights   []mat.Matrix
	biases    []mat.Matrix
}

//--- METHODS

// Backprop returns a tuple representing the gradient of the cost function `C_x`.
// 'biasesByLayer' and 'weightsByLayer' are layer-by-layer lists of matrices, similar to `Network.biases` and `Network.weights`.
func (net *Network) Backprop(x *Input) (biasesByLayer, weightsByLayer []mat.Matrix) {
	for _, b := range net.biases {
		r, c := b.Dims()
		data := make([]float64, r*c)
		biasesLayer := mat.NewDense(r, c, data)
		biasesByLayer = append(biasesByLayer, biasesLayer)
	}
	for _, w := range net.weights {
		r, c := w.Dims()
		data := make([]float64, r*c)
		weightsLayer := mat.NewDense(r, c, data)
		weightsByLayer = append(weightsByLayer, weightsLayer)
	}
	// Feedforward
	activations := []mat.Matrix{x.ToVector().T()}
	zs := []mat.Matrix{}
	activatn := activations[0]
	for i := 0; i < net.NumLayers()-1; i++ {
		z := matrix.Add(matrix.Dot(net.weights[i], activatn.T()).T(), net.biases[i])
		zs = append(zs, z)
		activatn = matrix.Apply(activation.Sigmoid, z)
		activations = append(activations, activatn)
	}
	// Backward pass
	delta := matrix.Multiply(net.CostDerivative(activations[len(activations)-1], x.Label.Vector), matrix.Apply(activation.SigmoidPrime, zs[len(zs)-1]))
	biasesByLayer[len(biasesByLayer)-1] = delta
	weightsByLayer[len(weightsByLayer)-1] = matrix.Dot(delta.T(), activations[len(activations)-2])
	if net.NumLayers() > 2 {
		for l := range utils.XRange(2, net.NumLayers()-1, 1) {
			z := zs[len(zs)-l]
			sp := matrix.Apply(activation.Sigmoid, z)
			delta = matrix.Multiply(matrix.Dot(delta, net.weights[len(net.weights)-l+1]), sp)
			biasesByLayer[len(biasesByLayer)-l] = delta
			weightsByLayer[len(weightsByLayer)-l] = matrix.Dot(delta.T(), activations[len(activations)-l-1])
		}
	}
	return
}

// CostDerivative returns the vector of partial derivatives `𝛿C_x / 𝛿a` for the output activations.
func (net *Network) CostDerivative(outputActivations mat.Matrix, y mat.Vector) mat.Matrix {
	return matrix.Subtract(outputActivations, y.T())
}

// Evaluate returns the number of test inputs for which the neural network outputs the correct result.
// Note that the neural network's output is assumed to be the index of whichever neuron in the final layer has the highest activation.
func (net *Network) Evaluate(test Dataset) (sum int) {
	for _, input := range test {
		testData := input.ToVector()
		output := net.FeedForward(testData)
		max := mat.Max(output)
		r, _ := output.Dims()
		col := mat.Col(nil, r, output)
		var result int
		for idx, val := range col {
			if val == max {
				result = idx
				break
			}
		}
		if result == int(math.Round(input.Label.Value)) {
			sum++
		}
	}
	return
}

// FeedForward returns the output of the network if `a` is input.
func (net *Network) FeedForward(a mat.Vector) (output mat.Matrix) {
	output = a.T()
	for i := 0; i < net.NumLayers()-1; i++ {
		// sigmoid(w·a + b)
		output = matrix.Apply(activation.Sigmoid, matrix.Add(matrix.Dot(net.weights[i], output.T()).T(), net.biases[i]))
	}
	return output
}

// SGD trains the neural network using mini-batch stochastic gradient descent.
// The 'training' dataset is a list of `Input` tuples representing the training data and the desired outputs.
// The other non-optional parameters are self-explanatory.
// If 'test' dataset is provided then the network will be evaluated against the test data after each epoch,
// and partial progress printed out. This is useful for tracking progress, but slows things down substantially.
func (net *Network) SGD(training Dataset, epochs int, miniBatchSize int, eta float64, test ...Dataset) {
	var (
		nTest int
		n     int
	)
	if len(test) > 0 {
		nTest = len(test[0])
	}
	n = len(training)
	for j := 0; j < epochs; j++ {
		training.Shuffle()
		var miniBatches []Dataset
		for k := range utils.XRange(0, n-miniBatchSize, miniBatchSize) {
			miniBatch := training[k : k+miniBatchSize]
			miniBatches = append(miniBatches, miniBatch)
		}
		for _, miniBatch := range miniBatches {
			net.UpdateMiniBatch(miniBatch, eta)
		}
		if len(test) > 0 {
			fmt.Printf("epoch %d: %d / %d\n", j+1, net.Evaluate(test[0]), nTest)
		} else {
			fmt.Printf("epoch %d complete\n", j+1)
		}
	}
}

// UpdateMiniBatch updates the network's weights and biases by applying gradient descent
// using backpropagation to a single mini batch.
// The 'miniBatch' is a list of `Inputs`, and 'eta' is the learning rate.
func (net *Network) UpdateMiniBatch(miniBatch Dataset, eta float64) {
	var biasesByLayer []mat.Matrix
	for _, b := range net.biases {
		r, c := b.Dims()
		data := make([]float64, r*c)
		biasesLayer := mat.NewDense(r, c, data)
		biasesByLayer = append(biasesByLayer, biasesLayer)
	}
	var weightsByLayer []mat.Matrix
	for _, w := range net.weights {
		r, c := w.Dims()
		data := make([]float64, r*c)
		weightsLayer := mat.NewDense(r, c, data)
		weightsByLayer = append(weightsByLayer, weightsLayer)
	}
	for _, input := range miniBatch {
		deltaBiasesByLayer, deltaWeightsByLayer := net.Backprop(input)
		for i, biases := range biasesByLayer {
			biasesByLayer[i] = matrix.Add(biases, deltaBiasesByLayer[i])
		}
		for i, weights := range weightsByLayer {
			weightsByLayer[i] = matrix.Add(weights, deltaWeightsByLayer[i])
		}
	}
	for i, biases := range net.biases {
		net.biases[i] = matrix.Subtract(biases, matrix.Scale(eta/float64(len(miniBatch)), biasesByLayer[i]))
	}
	for i, weights := range net.weights {
		net.weights[i] = matrix.Subtract(weights, matrix.Scale(eta/float64(len(miniBatch)), weightsByLayer[i]))
	}
}

// NumLayers is utility method returning the number of layers in the network.
func (net *Network) NumLayers() int {
	return net.numLayers
}

// OutputSize returns the size of the last layer.
func (net *Network) OutputSize() int {
	return net.Sizes[net.NumLayers()-1]
}

//--- FUNCTIONS

// Init ...
// The list `sizes` contains the number of neurons in the respective layers of the network.
// For example, if the list was [2, 3, 1] then it would be a three-layer network, with the
// first layer containing 2 neurons, the second layer 3 neurons, and the third layer 1 neuron.
// The biases and weights for the network are initialized randomly, using a Gaussian distribution
// with mean 0, and variance 1. Note that the first layer is assumed to be an input layer, and by
// convention we won’t set any biases for those neurons, since biases are only ever used in computing
// the outputs from later layers.
func Init(sizes []int) (n *Network, err error) {
	if len(sizes) < 2 {
		err = errors.New("not enough layers")
		return
	}

	// Biases
	biases := make([]mat.Matrix, len(sizes)-1)
	for i, size := range sizes[1:] {
		biases[i] = matrix.Random(1, size, 2)
	}

	// Weights
	tuples, err := utils.Zip(sizes[:len(sizes)-1], sizes[1:])
	if err != nil {
		return
	}
	weights := make([]mat.Matrix, len(tuples))
	for i, tuple := range tuples {
		weights[i] = matrix.Random(tuple.J, tuple.I, 2)
	}

	return &Network{
		Sizes:     sizes,
		numLayers: len(sizes),
		weights:   weights,
		biases:    biases,
	}, nil
}
