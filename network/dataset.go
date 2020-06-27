package network

import (
	"gonum.org/v1/gonum/mat"
)

//--- TYPES

// Dataset ...
type Dataset []Input

// Input ...
type Input struct {
	Data  []float64
	Label float64
}

//--- METHODS

// ToVector ...
func (i *Input) ToVector() mat.Vector {
	return mat.NewVecDense(len(i.Data), i.Data)
}
