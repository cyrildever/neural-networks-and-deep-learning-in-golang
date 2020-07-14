package network

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"neuraldeep/activation"
	"neuraldeep/cost"
	"neuraldeep/utils"
	"neuraldeep/utils/matrix"
	"os"

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

// Accuracy returns the number of inputs in 'data' for which the neural network outputs the correct result.
// The neural network's output is assumed to be the index of whichever neuron in the final layer has the highest activation.
func (net *Network2) Accuracy(data Dataset) (sum int) {
	for _, input := range data {
		x := input.ToVector()
		y := int(input.Label.Value)
		output := net.FeedForward(x)
		max := mat.Max(output)
		_, c := output.Dims()
		col := make([]float64, c)
		for i := 0; i < c; i++ {
			col[i] = output.At(0, i)
		}
		var result int
		for idx, val := range col {
			if val == max {
				result = idx
				break
			}
		}
		if result == y {
			sum++
		}
	}
	return
}

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
	delta := net.Cost.Delta(activations[len(activations)-1], x.Label.Vector, zs[len(zs)-1])
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
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	bytes, err := ioutil.ReadAll(f)
	if err != nil {
		return err
	}
	var n Network
	err = json.Unmarshal(bytes, &n)
	if err != nil {
		return err
	}
	var c cost.Cost
	switch n.Cost {
	case cost.CROSS_ENTROPY:
		c = cost.CrossEntropyCost{}
	case cost.QUADRATIC_COST:
		c = cost.QuadraticCost{}
	default:
		return errors.New("invalid cost function")
	}
	n2, err := Initial(n.Sizes, c)
	if err != nil {
		return err
	}
	for i, wData := range n.Weights {
		r, c := n2.weights[i].Dims()
		mW := mat.NewDense(r, c, wData)
		n2.weights[i] = mW
	}
	for i, bData := range n.Biases {
		r, c := n2.biases[i].Dims()
		mB := mat.NewDense(r, c, bData)
		n2.biases[i] = mB
	}
	net.Sizes = n2.Sizes
	net.Cost = n2.Cost
	net.numLayers = n2.NumLayers()
	net.weights = n2.weights
	net.biases = n2.biases
	return nil
}

// Save saves the neural network to the file 'path'.
func (net *Network2) Save(path string) error {
	var wList [][]float64
	for _, weights := range net.weights {
		var wL []float64
		r, c := weights.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				wL = append(wL, weights.At(i, j))
			}
		}
		wList = append(wList, wL)
	}
	var bList [][]float64
	for _, biases := range net.biases {
		var bL []float64
		r, c := biases.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				bL = append(bL, biases.At(i, j))
			}
		}
		bList = append(bList, bL)
	}
	data := Network{
		Sizes:   net.Sizes,
		Cost:    net.Cost.GetName(),
		Weights: wList,
		Biases:  bList,
	}
	jsonNetwork, err := json.Marshal(data)
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(path, jsonNetwork, 0644)
	if err != nil {
		return err
	}
	return nil
}

// SGD trains the neural network using mini-batch stochastic gradient descent.
// The 'trainingData' is a list of `Input` lines representing the training inputs and desired outputs.
// The other non-optional parameters are self-explanatory, as is the regularization parameter 'lambda'.
// The method also accepts 'evaluationData', usually either the validation or test data.
// We can monitor the cost and accuracy on either the evaluation data or the training data, by setting the appropriate flags.
// The method returns four lists: the (per-epoch) costs on the evaluation data, the accuracies on the evaluation data,
// the costs on the training data, and the accuracies on the training data. So, for example, if we train for 30 epochs,
// then the first list will be a 30-element list containing the cost on the evaluation data at the end of each epoch.
// Note that the lists are empty if the corresponding flag is not set. These flags are set as boolean values in the 'monitors' parameter
// in the following order: 'monitorEvaluationCost', 'monitorEvaluationAccuracy', 'monitorTrainingCost', 'monitorTrainingAccuracy'.
func (net *Network2) SGD(training Dataset, epochs, miniBatchSize int, eta, lambda float64, evaluation Dataset, monitors ...bool) (evaluationCost []float64, evaluationAccuracy []int, trainingCost []float64, trainingAccuracy []int) {
	var (
		nData, n                                                                                       int
		monitorEvaluationCost, monitorEvaluationAccuracy, monitorTrainingCost, monitorTrainingAccuracy bool
	)
	if len(evaluation) > 0 {
		nData = len(evaluation)
	}
	n = len(training)
	if len(monitors) > 0 {
		monitorEvaluationCost = monitors[0]
	}
	if len(monitors) > 1 {
		monitorEvaluationAccuracy = monitors[1]
	}
	if len(monitors) > 2 {
		monitorTrainingCost = monitors[2]
	}
	if len(monitors) == 4 {
		monitorTrainingAccuracy = monitors[3]
	}
	for j := 0; j < epochs; j++ {
		training.Shuffle()
		var miniBatches []Dataset
		for k := range utils.XRange(0, n-miniBatchSize, miniBatchSize) {
			miniBatch := training[k : k+miniBatchSize]
			miniBatches = append(miniBatches, miniBatch)
		}
		for _, miniBatch := range miniBatches {
			net.UpdateMiniBatch(miniBatch, eta, lambda, n)
		}
		fmt.Printf("epoch %d complete\n", j+1)
		if monitorTrainingCost {
			tc := net.TotalCost(training, lambda)
			trainingCost = append(trainingCost, tc)
			fmt.Printf("cost on training data: %.2f\n", tc)
		}
		if monitorTrainingAccuracy {
			ta := net.Accuracy(training)
			trainingAccuracy = append(trainingAccuracy, ta)
			fmt.Printf("accuracy on training data: %d / %d\n", ta, n)
		}
		if monitorEvaluationCost {
			ec := net.TotalCost(evaluation, lambda)
			evaluationCost = append(evaluationCost, ec)
			fmt.Printf("cost on evaluation data: %.2f\n", ec)
		}
		if monitorEvaluationAccuracy {
			ea := net.Accuracy(evaluation)
			evaluationAccuracy = append(evaluationAccuracy, ea)
			fmt.Printf("accuracy on evaluation data: %d / %d\n", ea, nData)
		}
		fmt.Println("")
	}
	return
}

// TotalCost returns the total cost for the data set 'data'.
func (net *Network2) TotalCost(data Dataset, lambda float64) (c float64) {
	for _, input := range data {
		x := input.ToVector()
		a := net.FeedForward(x)
		c += net.Cost.Function(a, input.Label.Vector) / float64(len(data))
	}
	sum := 0.0
	for _, w := range net.weights {
		sum += math.Pow(mat.Norm(w, 2), 2)
	}
	c += 0.5 * (lambda / float64(len(data))) * sum
	return
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
		costFunction, _ = cost.New(cost.CROSS_ENTROPY)
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
		bs[i] = matrix.Random(1, size, 2)
	}

	// Weights
	tuples, err := utils.Zip(sizes[:len(sizes)-1], sizes[1:])
	if err != nil {
		return
	}
	ws := make([]mat.Matrix, len(tuples))
	for i, tuple := range tuples {
		rand := matrix.Random(tuple.J, tuple.I, 2)
		ws[i] = matrix.Apply(func(i, j int, v float64) float64 {
			x := float64(tuple.I)
			if x < 0 {
				x = -x
			}
			return v / math.Sqrt(x)
		}, rand)
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
