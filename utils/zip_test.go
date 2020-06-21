package utils_test

import (
	"neuraldeep/utils"
	"testing"

	"gotest.tools/assert"
)

// TestZip ...
func TestZip(t *testing.T) {
	b := []int{1, 2}
	w := []int{3, 4}
	zipped, err := utils.Zip(b, w)
	if err != nil {
		t.Fatal(err)
	}
	assert.DeepEqual(t, zipped[0], utils.IntTuple{1, 3})
	assert.DeepEqual(t, zipped[1], utils.IntTuple{2, 4})
}
