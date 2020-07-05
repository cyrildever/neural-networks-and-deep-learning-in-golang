package cost

import (
	"neuraldeep/utils/matrix"

	"gonum.org/v1/gonum/mat"
)

//--- TYPES

// CrossEntropyCost ...
type CrossEntropyCost struct {
	A mat.Matrix
	Y mat.Vector
}

//--- METHODS

// Function returns the cost associated with an output `A` and desired output `Y`.
func (this CrossEntropyCost) Function() float64 {
	return mat.Sum(matrix.Subtract(matrix.Multiply(changeSign(this.Y.T()), matrix.Log(this.A)), matrix.Multiply(oneMinus(this.Y.T()), matrix.Log(oneMinus(this.A)))))
}

// Delta returns the error delta from the output layer.
// Note that the parameter `z` is not used by the method. It is included in the method's parameters
// in order to make the interface consistent with the delta method for other cost classes.
func (this CrossEntropyCost) Delta(z mat.Matrix) mat.Matrix {
	return matrix.Subtract(this.A, this.Y.T())
}

// Init ...
func (this CrossEntropyCost) Init(a mat.Matrix, y mat.Vector) {
	this.A = a
	this.Y = y
}

// utility functions

func changeSign(m mat.Matrix) mat.Matrix {
	return matrix.Apply(func(i, j int, v float64) float64 {
		return -v
	}, m)
}

func oneMinus(m mat.Matrix) mat.Matrix {
	return matrix.Apply(func(i, j int, v float64) float64 {
		return 1 - v
	}, m)
}
