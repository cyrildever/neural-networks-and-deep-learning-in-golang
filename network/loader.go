package network

import (
	"bufio"
	"encoding/csv"
	"io"
	"neuraldeep/utils"
	"os"
	"strconv"
)

const (
	sizeLine = 1 + 784 // label + image pixels
)

// LoadData mixes Michael Nielsen's load_data() and load_data_wrapper() functions into one through the use of the `Input` object.
// Another difference is that we aren't actually using slightly different formats for the training data and the validation / test data,
// ie. the `Input.Label` field will hold both the digital value of the classification and the 10-dimensional vector.
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
		input, e := readLine(record, 10)
		if e != nil {
			err = e
			return
		}
		if line < sizeTraining {
			// Training data
			training[line] = &input
		} else {
			// Validation data
			validation[line-sizeTraining] = &input
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
		input, e := readLine(record, 10)
		if e != nil {
			err = e
			return
		}
		test[line] = &input
		line++
	}
	return
}

func readLine(record []string, size int) (input Input, err error) {
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
		Label: ToLabel(data[0], size),
	}
	return
}
