package cost

import (
	"errors"

	"gonum.org/v1/gonum/mat"
)

// Cost ...
type Cost interface {
	Function(a mat.Matrix, y mat.Vector) float64
	Delta(a mat.Matrix, y mat.Vector, z mat.Matrix) mat.Matrix
	GetName() string
}

// New ...
func New(name string) (Cost, error) {
	switch name {
	case CROSS_ENTROPY:
		return CrossEntropyCost{Name: name}, nil
	case QUADRATIC_COST:
		return QuadraticCost{Name: name}, nil
	default:
		return nil, errors.New("unavailable cost function")
	}
}
