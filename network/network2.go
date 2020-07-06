package network

import (
	"errors"
	"math"
	"neuraldeep/activation"
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

// Backprop returns a tuple representing the gradient of the cost function `C_x`.
// 'biasesByLayer' and 'weightsByLayer' are layer-by-layer lists of matrices, similar to `Network.biases` and `Network.weights`.
func (net *Network2) Backprop(x *Input) (biasesByLayer, weightsByLayer []mat.Matrix) {
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
	net.Cost.Init(activations[len(activations)-1], x.Label.Vector)
	delta := net.Cost.Delta(zs[len(zs)-1])
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

// FeedForward returns the output of the network if `a` is input.
func (net *Network2) FeedForward(a mat.Vector) (output mat.Matrix) {
	output = a.T()
	for i := 0; i < net.NumLayers()-1; i++ {
		// sigmoid(w·a + b)
		output = matrix.Apply(activation.Sigmoid, matrix.Add(matrix.Dot(net.weights[i], output.T()).T(), net.biases[i]))
	}
	return output
}

// Load loads a neural network from the file 'path' into the current Network2 instance.
func (net *Network2) Load(path string) error {
	// TODO ######
	return nil
}

// UpdateMiniBatch updates the network's weights and biases by applying gradient descent
// using backpropagation to a single mini batch.
// The 'miniBatch' is a list of `Inputs`, 'eta' is the learning rate, 'lambda' is the
// regularization parameter, and 'n' is the total size of the training data set.
func (net *Network2) UpdateMiniBatch(miniBatch Dataset, eta, lambda float64, n int) {
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
		net.weights[i] = matrix.Subtract(matrix.Scale(1-eta*(lambda/float64(n)), weights), matrix.Scale(eta/float64(len(miniBatch)), weightsByLayer[i]))
	}
}

//---

// NumLayers is utility method returning the number of layers in the network.
func (net *Network2) NumLayers() int {
	return net.numLayers
}

// OutputSize returns the size of the last layer.
func (net *Network2) OutputSize() int {
	return net.Sizes[net.NumLayers()-1]
}

//--- FUNCTIONS

// Initial ...
// The list ``sizes`` contains the number of neurons in the respective layers of the network.
// For example, if the list was [2, 3, 1] then it would be a three-layer network, with the
// first layer containing 2 neurons, the second layer 3 neurons, and the third layer 1 neuron.
// The biases and weights for the network are initialized randomly, using DefaultWeightInitializer().
func Initial(sizes []int, fn ...cost.Cost) (n *Network2, err error) {
	if len(sizes) < 2 {
		err = errors.New("not enough layers")
		return
	}

	var costFunction cost.Cost
	if len(fn) != 1 {
		costFunction = cost.CrossEntropyCost{}
	} else {
		costFunction = fn[0]
	}

	biases, weights, err := DefaultWeightInitializer(sizes)
	if err != nil {
		return
	}

	return &Network2{
		Sizes:     sizes,
		Cost:      costFunction,
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
