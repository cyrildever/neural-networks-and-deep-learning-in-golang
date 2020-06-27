package network

import (
	"bufio"
	"encoding/csv"
	"io"
	"neuraldeep/utils"
	"os"
	"strconv"
)

// Input ...
type Input struct {
	Data  []float64
	Label float64
}

// Dataset ...
type Dataset []Input

// LoadData ...
func LoadData() (training Dataset, validation Dataset, test Dataset, err error) {
	_, err = utils.Unzip("./data/mnist_train.zip", "./data/loaded/")
	if err != nil {
		return
	}
	trainingFile, err := os.Open("./data/loaded/mnist_train.csv")
	if err != nil {
		return
	}
	defer trainingFile.Close()
	r1 := csv.NewReader(bufio.NewReader(trainingFile))
	sizeTraining := 50_000
	training = make(Dataset, sizeTraining)
	sizeValidation := 10_000
	validation = make(Dataset, sizeValidation)
	line := 0
	for {
		record, e := r1.Read()
		if e == io.EOF {
			break
		}
		input, e := readLine(record)
		if e != nil {
			err = e
			return
		}
		if line < sizeTraining {
			// Training data
			training[line] = input
		} else {
			// Validation data
			validation[line-sizeTraining] = input
		}
		line++
	}

	// Test data
	_, err = utils.Unzip("./data/mnist_test.zip", "./data/loaded/")
	if err != nil {
		return
	}
	testFile, err := os.Open("./data/loaded/mnist_test.csv")
	if err != nil {
		return
	}
	defer testFile.Close()
	r2 := csv.NewReader(bufio.NewReader(testFile))
	test = make(Dataset, 10_000)
	line = 0
	for {
		record, e := r2.Read()
		if e == io.EOF {
			break
		}
		input, e := readLine(record)
		if e != nil {
			err = e
			return
		}
		test[line] = input
		line++
	}
	return
}

const sizeLine = 1 + 784 // label + image pixels

func readLine(record []string) (input Input, err error) {
	data := make([]float64, sizeLine)
	for i := 0; i < sizeLine; i++ {
		d, e := strconv.ParseFloat(record[i], 64)
		if e != nil {
			err = e
			return
		}
		data[i] = d
	}
	input = Input{
		Data:  data[1:],
		Label: data[0],
	}
	return
}
