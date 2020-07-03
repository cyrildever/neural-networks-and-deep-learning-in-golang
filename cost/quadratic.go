package cost

import (
	"math"
	"neuraldeep/activation"
	"neuraldeep/utils/matrix"

	"gonum.org/v1/gonum/mat"
)

//--- TYPES

// QuadraticCost ...
type QuadraticCost struct {
	A mat.Matrix
	Y mat.Vector
}

//--- METHODS

// Function return the cost associated with an output `A` and desired output `Y`.
func (this *QuadraticCost) Function() float64 {
	return math.Pow(0.5*mat.Norm(matrix.Subtract(this.A, this.Y.T()), 2), 2)
}

// Delta returns the error delta from the output layer.
func (this *QuadraticCost) Delta(z mat.Matrix) mat.Matrix {
	return matrix.Multiply(matrix.Subtract(this.A, this.Y.T()), matrix.Apply(activation.SigmoidPrime, z))
}
