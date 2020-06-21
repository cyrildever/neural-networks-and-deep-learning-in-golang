package main

import (
	"flag"
	"fmt"
	"neuraldeep/network"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// Usage:
//
// $ ./neuraldeep -layers 2,3,1
func main() {
	operation := flag.String("o", "feedforward", "operation to proceed")
	layersStr := flag.String("layers", "", "comma-separated list of number of neurons per layer")
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
	fmt.Printf("nb of layers in the network: %d", n.NumLayers())

	// Forward propagation
	switch *operation {
	case "feedforward":
		a := mat.NewVecDense(10, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) // DEBUG
		output := n.FeedForward(a)
		fmt.Println(output) // DEBUG
	default:
		fmt.Println("invalid operation")
	}
}
