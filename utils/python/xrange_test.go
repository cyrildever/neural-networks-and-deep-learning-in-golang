package python_test

import (
	"neuraldeep/utils/python"
	"testing"

	"gotest.tools/assert"
)

// TestXRange ...
func TestXRange(t *testing.T) {
	expected := []int{0, 2, 4, 6, 8}
	var found []int
	for v := range python.XRange(0, 8, 2) {
		found = append(found, v)
	}
	assert.DeepEqual(t, found, expected)
}
