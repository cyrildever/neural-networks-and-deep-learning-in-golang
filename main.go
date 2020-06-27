package main

import (
	"errors"
	"flag"
	"fmt"
	"neuraldeep/network"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Usage:
//
// For one line of data:
// `$ ./neuraldeep --op predict --layers 6,50,20,3 --data "1,2.4,3,4,5,-6" --label 3`
//
// To use the MNIST dataset:
// `$ ./neuraldeep --op train --layers 784,30,10 --data training --useMNIST true`
func main() {
	// Parse command line arguments
	operation := flag.String("op", "", "operation to proceed: predict | test | train")
	layersStr := flag.String("layers", "", "comma-separated list of number of neurons per layer (the first one being the size of the input layer)")
	dataStr := flag.String("data", "", "a single data set to feed the first layer (a comma-separated list of float64), or the name of the MNIST set")
	labelStr := flag.String("label", "", "the label/target of the passed value as a float64 number")
	src := flag.String("src", "", "the source file to use as input data")
	useMNIST := flag.Bool("useMNIST", false, "set to true to use MNIST dataset (the layers flag should start with 784 and end with 10)")
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
	dataset := network.Dataset{}
	if *useMNIST {
		training, validation, test, err := network.LoadData()
		if err != nil {
			panic(err)
		}
		switch *dataStr {
		case "test":
			dataset = test
		case "training":
			dataset = training
		case "validation":
			dataset = validation
		}
	} else {
		if *dataStr != "" && *src == "" {
			// Read from command line
			dataArr := strings.Split(*dataStr, ",")
			input := network.Input{
				Data: make([]float64, len(dataArr)),
			}
			for _, d := range dataArr {
				data, err := strconv.ParseFloat(d, 64)
				if err != nil {
					panic(err)
				}
				input.Data = append(input.Data, data)
			}
			if *labelStr != "" {
				label, err := strconv.ParseFloat(*labelStr, 64)
				if err != nil {
					panic(err)
				}
				input.Label = label
			}
			dataset = append(dataset, input)
		} else if *src == "" {
			// Retrieve from file
			// TODO ####
			fmt.Println("Not implemented yet")
		}
	}

	// Process the operation
	t0 := time.Now()
	switch *operation {
	case "predict":
		fmt.Println("predicting...")
		if *useMNIST {
			// TODO ####
			elapsed := time.Since(t0)
			fmt.Printf("elapsed: %d ms\n", elapsed.Milliseconds())
		} else {
			a := mat.NewVecDense(len(dataset[0].Data), dataset[0].Data)
			output := n.FeedForward(a)
			_, c := output.Dims()
			if c != lastLayerSize {
				panic(errors.New("size mismatch in result"))
			}
			elapsed := time.Since(t0)
			fmt.Printf("elapsed: %d ms\n", elapsed.Milliseconds())
			fmt.Printf("target: %f\n", dataset[0].Label)
			for i := 0; i < c; i++ {
				fmt.Printf("output neuron #%d: %f\n", i+1, output.At(0, i))
			}
		}
	case "test":
		fmt.Println("testing...")
		sum := n.Evaluate(dataset)
		elapsed := time.Since(t0)
		fmt.Printf("elapsed: %d ms\n", elapsed.Milliseconds())
		fmt.Printf("nbOfCorrectResults: %d\n", sum)
	case "train":
		fmt.Println("training...")
		// TODO ####
		elapsed := time.Since(t0)
		fmt.Printf("elapsed: %d ms\n", elapsed.Milliseconds())
	default:
		fmt.Println("invalid operation: ", *operation)
	}
}
