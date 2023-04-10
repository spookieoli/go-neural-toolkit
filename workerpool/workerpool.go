package workerpool

// The Payload to send to the worker threads
type Payload struct {
	// The function to be executed
	F interface{}
	// The result of the function
	R interface{}
}

// The ResultSet is the result of the worker threads
type ResultSet struct {
	// The results of the worker threads
	Results []interface{}
}

// The WorkerPool is the pool with goroutines
// Note: Choose the best number of workers for your machine
type WorkerPool struct {
	// Number of Worker Threads (goroutines)
	NumWorkers int
	// Channel for the worker threads
	In chan Payload
	// Channel for the worker threads
	Out chan ResultSet
}

// NewWorkerPool creates a new worker pool
func NewWorkerPool(numWorkers int) *WorkerPool {
	return &WorkerPool{
		NumWorkers: numWorkers,
		In:         make(chan Payload),
		Out:        make(chan ResultSet),
	}
}
