package matrix

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// see https://sausheong.github.io/posts/how-to-build-a-simple-artificial-neural-network-with-go

// Add computes the addition of two matrices.
func Add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

// Apply applies a function to each elements of a matrix.
func Apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

// Dot computes the dot product between two matrices.
func Dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

// Log applies the natural logarithm element-wise on the passed Matrix.
func Log(m mat.Matrix) mat.Matrix {
	return Apply(func(i, j int, v float64) float64 {
		if v == 0 {
			return 0
		}
		return math.Log(v)
	}, m)
}

// Multiply multiplies two matrices together.
func Multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

// Random initializes a matrix of `r` rows and `c` columns with randomized values of mean `v`.
func Random(r, c int, v float64) mat.Matrix {
	boundary := 1 / math.Sqrt(v) // To get a variance of 1 around a mean of v
	dist := distuv.Uniform{
		Min: -boundary,
		Max: boundary,
	}
	data := make([]float64, r*c)
	for i := 0; i < r*c; i++ {
		data[i] = dist.Rand()
	}
	return mat.NewDense(r, c, data)
}

// Scale multiplies a matrix by a scalar.
func Scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

// Subtract computes the substraction of two matrices.
func Subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}
