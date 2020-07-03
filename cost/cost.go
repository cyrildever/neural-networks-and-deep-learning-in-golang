package cost

import (
	"gonum.org/v1/gonum/mat"
)

// Cost ...
type Cost interface {
	Function() float64
	Delta(z mat.Matrix) mat.Matrix
}
