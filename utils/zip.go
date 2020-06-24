package utils

import (
	"errors"
)

// IntTuple ...
type IntTuple struct {
	I, J int
}

// Zip proceed like Python's zip function
func Zip(a, b []int) ([]IntTuple, error) {
	if len(a) != len(b) {
		return nil, errors.New("arguments must be of same length")
	}
	r := make([]IntTuple, len(a), len(b))
	for i, e := range a {
		r[i] = IntTuple{e, b[i]}
	}
	return r, nil
}
