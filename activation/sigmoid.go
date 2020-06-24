package activation

import (
	"math"
)

// Sigmoid is the sigmoid function.
func Sigmoid(i, j int, z float64) float64 {
	return 1 / (1 + math.Exp(-1*z))
}

// SigmoidPrime returns the derivative of the sigmoid function.
func SigmoidPrime(i, j int, z float64) float64 {
	return Sigmoid(i, j, z) * (1 - Sigmoid(i, j, z))
}
