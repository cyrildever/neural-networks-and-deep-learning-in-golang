package network

// Network is the JSON representation of the Networks.
type Network struct {
	Sizes   []int       `json:"sizes"`
	Cost    string      `json:"cost,omitempty"`
	Weights [][]float64 `json:"weights"`
	Biases  [][]float64 `json:"biases"`
}
