package network

import (
	"errors"
	"math"
	"neuraldeep/activation"
	"neuraldeep/utils"
	"neuraldeep/utils/matrix"

	"gonum.org/v1/gonum/mat"
)

//--- TYPES

// Network defines the neural network structure by instantiating it with an array of sizes,
// ie. the number of neurons per layer
type Network struct {
	Sizes     []int
	numLayers int
	weights   []mat.Matrix
	biases    []mat.Matrix
}

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

//--- METHODS

// Evaluate returns the number of test inputs for which the neural network outputs the correct result.
// Note that the neural network's output is assumed to be the index of whichever neuron in the final layer has the highest activation.
func (n *Network) Evaluate(test Dataset) (sum int) {
	for _, input := range test {
		a := mat.NewVecDense(len(input.Data), input.Data)
		output := n.FeedForward(a)
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
		if result == int(math.Round(input.Label)) {
			sum++
		}
	}
	return
}

// FeedForward returns the output of the network if `a` is input.
func (n *Network) FeedForward(a mat.Vector) (output mat.Matrix) {
	output = a.T()
	for i := 0; i < n.NumLayers()-1; i++ {
		// sigmoid(w·a + b)
		output = matrix.Apply(activation.Sigmoid, matrix.Add(matrix.Dot(n.weights[i], output.T()).T(), n.biases[i]))
	}
	return output
}

// NumLayers returns the number of layers in the network.
func (n *Network) NumLayers() int {
	return n.numLayers
}
