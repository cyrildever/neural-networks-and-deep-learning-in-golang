package main

import (
	"errors"
	"flag"
	"fmt"
	"neuraldeep/network"
	"strconv"
	"strings"
	"time"
)

// Usage:
//
// For one line of data:
// `$ ./neuraldeep --op=predict --layers=6,50,20,3 --data="1,2.4,3,4,5,-6" --label=3`
//
// To use the MNIST dataset:
// `$ ./neuraldeep --op=train --layers=784,30,10 --data=training --useMNIST=true --epochs=30 --size=10 --eta=3.0 --load=false`
func main() {
	// Parse command line arguments
	operation := flag.String("op", "", "operation to proceed: predict | test | train")
	layersStr := flag.String("layers", "", "comma-separated list of number of neurons per layer (the first one being the size of the input layer)")
	dataStr := flag.String("data", "", "a single data set to feed the first layer (a comma-separated list of float64), or the name of the MNIST set (test | training | validation)")
	labelStr := flag.String("label", "", "the label/target of the passed value as a float64 number")
	src := flag.String("src", "", "the source file to use as input data")
	useMNIST := flag.Bool("useMNIST", false, "set to true to use MNIST dataset (the layers flag should start with 784 and end with 10)")
	epochs := flag.Int("epochs", 1, "number of epochs")
	miniBatchSize := flag.Int("size", 10, "mini-batch size")
	eta := flag.Float64("eta", 0.1, "learning rate")
	load := flag.Bool("load", false, "set to `true` if you want to load an existing network")
	pathToExisting := flag.String("path", "./data/saved/network/", "path to the existing file")

	flag.Parse()

	fmt.Printf("Command to execute: $ ./neuraldeep --op=%s --layers=%s --data=%s --label=%s --src=%s --useMNIST=%t --epochs=%d --size=%d --eta=%f --load=%t\n===\n",
		*operation, *layersStr, *dataStr, *labelStr, *src, *useMNIST, *epochs, *miniBatchSize, *eta, *load)
	t0 := time.Now()

	// Initialize the network
	var net *network.Network
	layers := strings.Split(*layersStr, ",")
	var sizes []int
	for _, layer := range layers {
		size, err := strconv.Atoi(layer)
		if err != nil {
			panic(err)
		}
		sizes = append(sizes, size)
	}
	if *load {
		fmt.Println("loading from", *pathToExisting)
		n, err := network.Init(sizes)
		if err != nil {
			panic(err)
		}
		if err := network.Load(n, *pathToExisting); err != nil {
			panic(err)
		}
		net = n
	} else {
		n, err := network.Init(sizes)
		if err != nil {
			panic(err)
		}
		net = n
	}
	lastLayerSize := sizes[len(sizes)-1]
	fmt.Printf("network ready [nbOfLayers=%d, outputSize=%d]\n", net.NumLayers(), lastLayerSize)

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
				input.Label = network.ToLabel(label, net.OutputSize())
			}
			dataset = append(dataset, &input)
		} else if *src == "" {
			// Retrieve from file
			// TODO ####
			fmt.Println("Not implemented yet")
		}
	}

	// Process the operation
	t1 := time.Now()
	switch *operation {
	case "predict":
		fmt.Println("predicting...")
		if *useMNIST {
			// TODO ####
			elapsed := time.Since(t1)
			fmt.Printf("elapsed: %d ms\n", elapsed.Milliseconds())
		} else {
			a := dataset[0].ToVector()
			output := net.FeedForward(a)
			_, c := output.Dims()
			if c != lastLayerSize {
				panic(errors.New("size mismatch in result"))
			}
			elapsed := time.Since(t1)
			fmt.Printf("elapsed: %d ms\n", elapsed.Milliseconds())
			fmt.Printf("target: %f\n", dataset[0].Label)
			for i := 0; i < c; i++ {
				fmt.Printf("output neuron #%d: %f\n", i+1, output.At(0, i))
			}
		}
	case "test":
		fmt.Println("testing...")
		sum := net.Evaluate(dataset)
		elapsed := time.Since(t1)
		fmt.Printf("elapsed: %d ms\n", elapsed.Milliseconds())
		fmt.Printf("nbOfCorrectResults: %d\n", sum)
	case "train":
		fmt.Println("training...")
		net.SGD(dataset, *epochs, *miniBatchSize, *eta)
		elapsed := time.Since(t1)
		fmt.Printf("elapsed: %d ms\n", elapsed.Milliseconds())
		fmt.Println("saving to ./data/saved/network/")
		if err := net.Save(); err != nil {
			panic(err)
		}
		elapsed = time.Since(t0)
		fmt.Printf("terminated in %f s\n", elapsed.Seconds())
	default:
		fmt.Println("invalid operation: ", *operation)
	}
}
