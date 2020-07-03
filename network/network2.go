package network

import (
	"errors"
	"math"
	"neuraldeep/cost"
	"neuraldeep/utils"
	"neuraldeep/utils/matrix"

	"gonum.org/v1/gonum/mat"
)

//--- TYPES

// Network2 ...
type Network2 struct {
	Sizes     []int
	Cost      cost.Cost
	numLayers int
	weights   []mat.Matrix
	biases    []mat.Matrix
}

//--- METHODS

//--- FUNCTIONS

// Initial ...
// The list ``sizes`` contains the number of neurons in the respective layers of the network.
// For example, if the list was [2, 3, 1] then it would be a three-layer network, with the
// first layer containing 2 neurons, the second layer 3 neurons, and the third layer 1 neuron.
// The biases and weights for the network are initialized randomly, using DefaultWeightInitializer().
func Initial(sizes []int, fn cost.Cost) (n *Network2, err error) {
	if len(sizes) < 2 {
		err = errors.New("not enough layers")
		return
	}

	biases, weights, err := DefaultWeightInitializer(sizes)
	if err != nil {
		return
	}

	return &Network2{
		Sizes:     sizes,
		Cost:      fn,
		numLayers: len(sizes),
		weights:   weights,
		biases:    biases,
	}, nil
}

// Initialize each weight using a Gaussian distribution with mean 0 and standard deviation 1
// over the square root of the number of weights connecting to the same neuron.
// Initialize the biases using a Gaussian distribution with mean 0 and standard deviation 1.
// Note that the first layer is assumed to be an input layer, and by convention we won't set
// any biases for those neurons, since biases are only ever used in computing the outputs from later layers.
func DefaultWeightInitializer(sizes []int) (biases, weights []mat.Matrix, err error) {
	// Biases
	bs := make([]mat.Matrix, len(sizes)-1)
	for i, size := range sizes[1:] {
		bs[i] = matrix.Apply(func(i, j int, v float64) float64 {
			return math.Sqrt(v)
		}, matrix.Random(1, size, 2))
	}

	// Weights
	tuples, err := utils.Zip(sizes[:len(sizes)-1], sizes[1:])
	if err != nil {
		return
	}
	ws := make([]mat.Matrix, len(tuples))
	for i, tuple := range tuples {
		ws[i] = matrix.Random(tuple.J, tuple.I, 2)
	}
	return bs, ws, nil
}

// Initialize the weights using a Gaussian distribution with mean 0 and standard deviation 1.
// Initialize the biases using a Gaussian distribution with mean 0 and standard deviation 1.
// Note that the first layer is assumed to be an input layer, and by convention we won't set any biases
// for those neurons, since biases are only ever used in computing the outputs from later layers.
// This weight and bias initializer uses the same approach as in Chapter 1, and is included for purposes of comparison.
// It will usually be better to use the default weight initializer instead.
func LargeWeightInitializer(sizes []int) (biases, weights []mat.Matrix, err error) {
	// Biases
	bs := make([]mat.Matrix, len(sizes)-1)
	for i, size := range sizes[1:] {
		bs[i] = matrix.Random(1, size, 2)
	}

	// Weights
	tuples, err := utils.Zip(sizes[:len(sizes)-1], sizes[1:])
	if err != nil {
		return
	}
	ws := make([]mat.Matrix, len(tuples))
	for i, tuple := range tuples {
		ws[i] = matrix.Random(tuple.J, tuple.I, 2)
	}
	return bs, ws, nil
}
