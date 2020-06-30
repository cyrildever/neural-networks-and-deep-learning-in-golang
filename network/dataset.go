package network

import (
	"math"
	"math/rand"
	"neuraldeep/utils"
	"time"

	"gonum.org/v1/gonum/mat"
)

//--- TYPES

// Dataset ...
type Dataset []*Input

// Input ...
type Input struct {
	Data  []float64
	Label *Label
}

// Label ...
type Label struct {
	Value  float64
	Vector mat.Vector
}

//--- METHODS

// Shuffle ...
func (ds Dataset) Shuffle() {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	for i := len(ds) - 1; i > 0; i-- {
		j := r.Intn(i + 1)
		ds[i], ds[j] = ds[j], ds[i]
	}
}

// ToVector ...
func (i *Input) ToVector() mat.Vector {
	return mat.NewVecDense(len(i.Data), i.Data)
}

//--- FUNCTIONS

// ToLabel returns the Label populating the vector assuming that the output layer represents all the possible output values,
// eg. if it's of size 10 then it means that the label could be any value from 0 to 9.
func ToLabel(value float64, size int) *Label {
	data := make([]float64, size)
	for i := range utils.XRange(0, size-1, 1) {
		if int(math.Round(value)) == i {
			data[i] = 1.0
		} else {
			data[i] = 0.0
		}
	}
	return &Label{
		Value:  value,
		Vector: mat.NewVecDense(size, data),
	}
}
