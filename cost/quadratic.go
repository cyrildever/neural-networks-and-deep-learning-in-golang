package cost

import (
	"math"
	"neuraldeep/activation"
	"neuraldeep/utils/matrix"

	"gonum.org/v1/gonum/mat"
)

const QUADRATIC_COST = "Quadratic"

//--- TYPES

// QuadraticCost ...
type QuadraticCost struct {
	Name string
}

//--- METHODS

// Function return the cost associated with an output `a` and desired output `y`.
func (q QuadraticCost) Function(a mat.Matrix, y mat.Vector) float64 {
	return math.Pow(0.5*mat.Norm(matrix.Subtract(a, y.T()), 2), 2)
}

// Delta returns the error delta from the output layer.
func (q QuadraticCost) Delta(a mat.Matrix, y mat.Vector, z mat.Matrix) mat.Matrix {
	return matrix.Multiply(matrix.Subtract(a, y.T()), matrix.Apply(activation.SigmoidPrime, z))
}

// GetName ...
func (q QuadraticCost) GetName() string {
	return q.Name
}
