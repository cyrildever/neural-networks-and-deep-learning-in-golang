package cost

import (
	"gonum.org/v1/gonum/mat"
)

// Cost ...
type Cost interface {
	Function(a mat.Matrix, y mat.Vector) float64
	Delta(a mat.Matrix, y mat.Vector, z mat.Matrix) mat.Matrix
	GetName() string
}
