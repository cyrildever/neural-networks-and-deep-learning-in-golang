package cost_test

import (
	"fmt"
	"math"
	"neuraldeep/cost"
	"testing"

	"gonum.org/v1/gonum/mat"
	"gotest.tools/assert"
)

// TestCrossEntropy ...
func TestCrossEntropy(t *testing.T) {
	a := mat.NewDense(1, 2, []float64{0.1, 0.2})
	y := mat.NewVecDense(2, []float64{0.3, 0.4})
	ce, err := cost.New(cost.CROSS_ENTROPY)
	if err != nil {
		t.Fatal(err)
	}

	// r = ∑ -y*Log(a)-(1-y)*Log(1-a)
	r1 := -0.3*math.Log(0.1) - (1-0.3)*math.Log(1-0.1)
	r2 := -0.4*math.Log(0.2) - (1-0.4)*math.Log(1-0.2)
	r := r1 + r2

	result := ce.Function(a, y)
	assert.Equal(t, result, r)

	delta := mat.NewDense(1, 2, []float64{0.1 - 0.3, 0.2 - 0.4})
	z := mat.NewDense(1, 2, []float64{0, 0})
	d := ce.Delta(a, y, z)
	for i := 0; i < 1; i++ {
		for j := 0; j < 2; j++ {
			assert.Equal(t, fmt.Sprintf("%.2f", d.At(i, j)), fmt.Sprintf("%.2f", delta.At(i, j)))
		}
	}
}

// TestQuadratic ...
func TestQuadratic(t *testing.T) {
	a := mat.NewDense(1, 2, []float64{0.1, 0.2})
	y := mat.NewVecDense(2, []float64{0.3, 0.4})
	q, err := cost.New(cost.QUADRATIC_COST)
	if err != nil {
		t.Fatal(err)
	}

	// r = 0.5 * ‖ a - y ‖^2
	r := 0.5 * math.Pow(mat.Norm(mat.NewDense(1, 2, []float64{0.1 - 0.3, 0.2 - 0.4}), 2), 2)

	result := q.Function(a, y)
	assert.Equal(t, fmt.Sprintf("%.2f", result), fmt.Sprintf("%.2f", r))
}
