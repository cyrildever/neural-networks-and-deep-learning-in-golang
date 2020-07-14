package main

import (
	"errors"
	"flag"
	"fmt"
	"neuraldeep/cost"
	"neuraldeep/network"
	"strconv"
	"strings"
	"time"
)

// Usage:
//
// For one line of data:
// `$ ./neuraldeep -n=1 -op=predict -layers=6,50,20,3 -data="1,2.4,3,4,5,-6" -label=3`
//
// To use the MNIST dataset:
// `$ ./neuraldeep -n=1 -op=train -layers=784,300,10 -data=training -useMNIST=true -epochs=30 -size=10 -eta=3.0 -load=false -eval=true`
// `$ ./neuraldeep -n=1 -op=test -layers=784,300,10 -data=test -useMNIST=true -load=true`
// `& ./neuraldeep -n=1 -op=predict -layers=784,300,10 -label=5 -data="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,18,18,18,126,136,175,26,166,255,247,127,0,0,0,0,0,0,0,0,0,0,0,0,30,36,94,154,170,253,253,253,253,253,225,172,253,242,195,64,0,0,0,0,0,0,0,0,0,0,0,49,238,253,253,253,253,253,253,253,253,251,93,82,82,56,39,0,0,0,0,0,0,0,0,0,0,0,0,18,219,253,253,253,253,253,198,182,247,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,156,107,253,253,205,11,0,43,154,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,1,154,253,90,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,139,253,190,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,190,253,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,241,225,160,108,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,81,240,253,253,119,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,186,253,253,150,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,93,252,253,187,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,249,253,249,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,46,130,183,253,253,207,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,39,148,229,253,253,253,250,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,114,221,253,253,253,253,201,78,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,66,213,253,253,253,253,198,81,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,171,219,253,253,253,253,195,80,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,172,226,253,253,253,253,244,133,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,136,253,253,253,212,135,132,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" -load=true`
//
// `$ ./neuraldeep -n=2 -op=train -cost=crossEntropy -layers=784,300,10 -data=training -useMNIST=true -epochs=30 -size=10 -eta=0.12 -lambda=5.0 -eval=true -load=false`
// `$ ./neuraldeep -n=2 -op=predict -cost=crossEntropy -layers=784,300,10 -data="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,18,18,18,126,136,175,26,166,255,247,127,0,0,0,0,0,0,0,0,0,0,0,0,30,36,94,154,170,253,253,253,253,253,225,172,253,242,195,64,0,0,0,0,0,0,0,0,0,0,0,49,238,253,253,253,253,253,253,253,253,251,93,82,82,56,39,0,0,0,0,0,0,0,0,0,0,0,0,18,219,253,253,253,253,253,198,182,247,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,156,107,253,253,205,11,0,43,154,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,1,154,253,90,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,139,253,190,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,190,253,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,241,225,160,108,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,81,240,253,253,119,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,186,253,253,150,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,93,252,253,187,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,249,253,249,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,46,130,183,253,253,207,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,39,148,229,253,253,253,250,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,114,221,253,253,253,253,201,78,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,66,213,253,253,253,253,198,81,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,171,219,253,253,253,253,195,80,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,172,226,253,253,253,253,244,133,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,136,253,253,253,212,135,132,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" -load=true -path="./data/saved/network2.json"`
func main() {
	// Parse command line arguments
	n := flag.String("n", "1", "the network implementation to use: 1 | 2 | 3")
	operation := flag.String("op", "", "operation to proceed: predict | test | train")
	layersStr := flag.String("layers", "", "comma-separated list of number of neurons per layer (the first one being the size of the input layer)")
	dataStr := flag.String("data", "", "a single data set to feed the first layer (a comma-separated list of float64), or the name of the MNIST set (test | training | validation)")
	labelStr := flag.String("label", "", "the label/target of the passed value as a float64 number")
	src := flag.String("src", "", "the source file to use as input data")
	useMNIST := flag.Bool("mnist", false, "set to true to use MNIST dataset (the layers flag should start with 784 and end with 10)")
	epochs := flag.Int("epochs", 1, "number of epochs")
	miniBatchSize := flag.Int("size", 10, "mini-batch size")
	eta := flag.Float64("eta", 0.1, "learning rate")
	load := flag.Bool("load", false, "set to `true` if you want to load an existing network")
	pathToExisting := flag.String("path", "./data/saved/network/", "path to the existing file")
	evaluate := flag.Bool("eval", false, "set to `true` to add evaluation at each training epoch")
	costFunction := flag.String("cost", "crossEntropy", "cost function: crossEntropy | quadratic")
	lambda := flag.Float64("lambda", 0.0, "the regularization parameter")

	flag.Parse()

	fmt.Printf("command to execute: $ ./neuraldeep -n=%s -op=%s -layers=%s -data=%s -label=%s -src=%s -mnist=%t -epochs=%d -size=%d -eta=%f -eval=%t -cost=%s -lambda=%f -load=%t -path=%s\n===\n",
		*n, *operation, *layersStr, *dataStr, *labelStr, *src, *useMNIST, *epochs, *miniBatchSize, *eta, *evaluate, *costFunction, *lambda, *load, *pathToExisting)
	t0 := time.Now()

	// Choose the implementation
	if *n == "1" {
		// NETWORK.PY ###

		// Initialize the network
		var net *network.Network1
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
		fmt.Printf("network %s ready [nbOfLayers=%d, outputSize=%d]\n", *n, net.NumLayers(), lastLayerSize)

		// Get the input data
		dataset := network.Dataset{}
		evalset := network.Dataset{}
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
				evalset = test
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
				for i, d := range dataArr {
					data, err := strconv.ParseFloat(d, 64)
					if err != nil {
						panic(err)
					}
					input.Data[i] = data
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
				fmt.Printf("target: #%d\n", int(dataset[0].Label.Value))
				for i := 0; i < c; i++ {
					fmt.Printf("output #%d: %f\n", i, output.At(0, i))
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
			if *evaluate {
				net.SGD(dataset, *epochs, *miniBatchSize, *eta, evalset)
			} else {
				net.SGD(dataset, *epochs, *miniBatchSize, *eta)
			}
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
	} else if *n == "2" {
		// NETWORK2.PY ###

		// Initialize the network
		var net *network.Network2
		layers := strings.Split(*layersStr, ",")
		var sizes []int
		for _, layer := range layers {
			size, err := strconv.Atoi(layer)
			if err != nil {
				panic(err)
			}
			sizes = append(sizes, size)
		}
		var cf cost.Cost
		switch *costFunction {
		case cost.CROSS_ENTROPY:
			cf, _ = cost.New(cost.CROSS_ENTROPY)
		case cost.QUADRATIC_COST:
			cf, _ = cost.New(cost.QUADRATIC_COST)
		default:
			panic("invalid cost function")
		}
		if *load {
			fmt.Printf("loading from %s\n", *pathToExisting)
			n, err := network.Initial(sizes, cf)
			if err != nil {
				panic(err)
			}
			if err := n.Load(*pathToExisting); err != nil {
				panic(err)
			}
			net = n
		} else {
			n, err := network.Initial(sizes, cf)
			if err != nil {
				panic(err)
			}
			net = n
		}
		lastLayerSize := sizes[len(sizes)-1]
		fmt.Printf("network %s ready [nbOfLayers=%d, outputSize=%d]\n", *n, net.NumLayers(), lastLayerSize)

		// Get the input data
		dataset := network.Dataset{}
		evalset := network.Dataset{}
		if *useMNIST {
			training, validation, test, err := network.LoadData()
			if err != nil {
				panic(err)
			}
			switch *dataStr {
			case "test":
				dataset = test
				evalset = validation
			case "training":
				dataset = training
				evalset = test
			case "validation":
				dataset = validation
				evalset = test
			}
		} else {
			if *dataStr != "" && *src == "" {
				// Read from command line
				dataArr := strings.Split(*dataStr, ",")
				input := network.Input{
					Data: make([]float64, len(dataArr)),
				}
				for i, d := range dataArr {
					data, err := strconv.ParseFloat(d, 64)
					if err != nil {
						panic(err)
					}
					input.Data[i] = data
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
				fmt.Printf("target: #%d\n", int(dataset[0].Label.Value))
				for i := 0; i < c; i++ {
					fmt.Printf("output #%d: %f\n", i, output.At(0, i))
				}
			}
		case "test":
			fmt.Println("Not implemented")
		case "train":
			fmt.Println("training...")
			if *evaluate {
				net.SGD(dataset, *epochs, *miniBatchSize, *eta, *lambda, evalset, true, true, true, true)
			} else {
				net.SGD(dataset, *epochs, *miniBatchSize, *eta, *lambda, network.Dataset{})
			}
			elapsed := time.Since(t1)
			fmt.Printf("elapsed: %d ms\n", elapsed.Milliseconds())
			fmt.Println("saving to ./data/saved/network2.json")
			if err := net.Save("./data/saved/network2.json"); err != nil {
				panic(err)
			}
			elapsed = time.Since(t0)
			fmt.Printf("terminated in %f s\n", elapsed.Seconds())
		default:
			fmt.Println("invalid operation: ", *operation)
		}
	} else {
		fmt.Println("not implemented yet")
	}
}
