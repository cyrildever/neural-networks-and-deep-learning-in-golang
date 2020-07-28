package python_test

import (
	"neuraldeep/utils/python"
	"testing"

	"gotest.tools/assert"
)

// TestZip ...
func TestZip(t *testing.T) {
	b := []int{1, 2}
	w := []int{3, 4}
	zipped, err := python.Zip(b, w)
	if err != nil {
		t.Fatal(err)
	}
	assert.DeepEqual(t, zipped[0], python.IntTuple{1, 3})
	assert.DeepEqual(t, zipped[1], python.IntTuple{2, 4})
}
