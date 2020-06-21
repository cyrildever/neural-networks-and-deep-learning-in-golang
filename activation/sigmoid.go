package activation

import (
	"math"
)

// Sigmoid is the sigmoid function.
func Sigmoid(i, j int, z float64) float64 {
	return 1 / (1 + math.Exp(-1*z))
}
