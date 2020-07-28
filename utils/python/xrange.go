package python

// XRange mimics Python's xrange providing a start index, a potentially included end index and an increment.
func XRange(start, end, increment int) <-chan int {
	c := make(chan int)
	go func() {
		for i := start; i <= end; i += increment {
			c <- i
		}
		close(c)
	}()
	return c
}
