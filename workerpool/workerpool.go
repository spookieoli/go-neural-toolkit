package workerpool

import "sync"

// The Payload to send to the worker threads
type Payload struct {
	// The function to be executed
	F func(...any) any
	// The Data to be sent to the functions
	D []interface{}
	//a waitgroup pointer to know when all the work is done
	Wg *sync.WaitGroup
}

// The ResultSet is the result of the worker threads
type ResultSet struct {
	// The results of the worker threads
	Results interface{}
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
	// Create the chans
	wp := &WorkerPool{
		NumWorkers: numWorkers,
		In:         make(chan Payload),
		Out:        make(chan ResultSet),
	}

	// Start the worker threads
	for i := 0; i < numWorkers; i++ {
		go func() {
			for {
				// Get the payload
				payload := <-wp.In
				// Execute the functions on the data
				result := payload.F(payload.D...)
				// Send the result back
				wp.Out <- ResultSet{Results: result}
				// Signal that the work is done
				payload.Wg.Done()
			}
		}()
	}

	// return the workerpool
	return wp
}
