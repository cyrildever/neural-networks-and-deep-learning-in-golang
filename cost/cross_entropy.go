package cost

import (
	"neuraldeep/utils/matrix"

	"gonum.org/v1/gonum/mat"
)

const CROSS_ENTROPY = "CrossEntropy"

//--- TYPES

// CrossEntropyCost ...
type CrossEntropyCost struct {
	Name string
}

//--- METHODS

// Function returns the cost associated with an output `a` and desired output `y`.
func (c CrossEntropyCost) Function(a mat.Matrix, y mat.Vector) float64 {
	return mat.Sum(matrix.Subtract(matrix.Multiply(changeSign(y.T()), matrix.Log(a)), matrix.Multiply(oneMinus(y.T()), matrix.Log(oneMinus(a)))))
}

// Delta returns the error delta from the output layer.
// Note that the parameter `z` is not used by the method. It is included in the method's parameters
// in order to make the interface consistent with the delta method for other cost classes.
func (c CrossEntropyCost) Delta(a mat.Matrix, y mat.Vector, z mat.Matrix) mat.Matrix {
	return matrix.Subtract(a, y.T())
}

// GetName ...
func (c CrossEntropyCost) GetName() string {
	return c.Name
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
