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

func NewModel(layers []layer.Layer, worker int) (*Model, error) {
	// Guard clause
	if len(layers) == 0 {
		return nil, fmt.Errorf("No Layers given")
	}
	// TODO: More Checks - eg: every Model needs an input layer etc.
	return &Model{
		Layers:     layers,
		Workerpool: workerpool.NewWorkerPool(worker),
	}, nil
}
