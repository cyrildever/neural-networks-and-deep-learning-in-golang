package main

import (
	"errors"
	"flag"
	"fmt"
	"neuraldeep/network"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// Usage:
//
// $ ./neuraldeep --layers 6,50,20,3 --data "1,2.4,3,4,5,-6"
func main() {
	// Parse command line arguments
	operation := flag.String("o", "feedforward", "operation to proceed")
	layersStr := flag.String("layers", "", "comma-separated list of number of neurons per layer (the first one being the size of the input layer)")
	dataStr := flag.String("data", "", "a single data set to feed the first layer (float64 number in a comma-separated list of size of the input layer")
	src := flag.String("src", "", "the source file to use as input data")
	flag.Parse()

	// Initialize the network
	layers := strings.Split(*layersStr, ",")
	var sizes []int
	for _, layer := range layers {
		size, err := strconv.Atoi(layer)
		if err != nil {
			panic(err)
		}
		sizes = append(sizes, size)
	}
	n, err := network.Init(sizes)
	if err != nil {
		panic(err)
	}
	lastLayerSize := sizes[len(sizes)-1]
	fmt.Printf("network ready [nbOfLayers=%d, outputSize=%d]\n", n.NumLayers(), lastLayerSize)

	// Get the input data
	var inputs []float64
	if *dataStr != "" && *src == "" {
		// Read from command line
		data := strings.Split(*dataStr, ",")
		for _, d := range data {
			input, err := strconv.ParseFloat(d, 64)
			if err != nil {
				panic(err)
			}
			inputs = append(inputs, input)
		}
	} else if *src == "" {
		// Retrieve from file
		// TODO ####
	}

	// Process the operation
	switch *operation {
	case "feedforward":
		// Forward propagation
		a := mat.NewVecDense(len(inputs), inputs)
		output := n.FeedForward(a)
		_, c := output.Dims()
		if c != lastLayerSize {
			panic(errors.New("size mismatch in result"))
		}
		for i := 0; i < c; i++ {
			fmt.Printf("output neuron #%d: %f\n", i+1, output.At(0, i))
		}
	default:
		fmt.Println("invalid operation")
	}
}
