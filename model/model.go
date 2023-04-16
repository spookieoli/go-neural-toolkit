package model

import (
	"fmt"
	"go-neural-toolkit/layer"
	"go-neural-toolkit/tensor"
	"go-neural-toolkit/workerpool"
)

type Model struct {
	// The layers of the model.
	Layers     []layer.Layer
	Workerpool *workerpool.WorkerPool
	Output     tensor.Tensor
}

// Check if the given layers are of Type InputLayer
func CheckIsInputLayer(layers []layer.Layer) bool {
	for _, l := range layers {
		if _, ok := l.(*layer.InputLayer); !ok {
			return false
		}
	}
	return true
}

// NewModel creates a new model. NewModel takes only the Input Layers of the Model and the number of worker threads.
func NewModel(layers []layer.Layer, worker int) (*Model, error) {
	// Guard clause
	if len(layers) == 0 {
		return nil, fmt.Errorf("No Layers given")
	}
	// Check if all Layers given are Inputlayers
	if CheckIsInputLayer(layers) == false {
		return nil, fmt.Errorf("No Input Layer given")
	}
	// Create the Model
	return &Model{
		Layers:     layers,
		Workerpool: workerpool.NewWorkerPool(worker),
	}, nil
}
