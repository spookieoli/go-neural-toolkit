package layer

import (
	"go-neural-toolkit/tensor"
	"go-neural-toolkit/workerpool"
)

// The Layer interface is the base interface for all layers.
type Layer interface {
	// GetName returns the name of the layer.
	GetName() string
	// GetWeights returns the weights of the layer.
	GetWeights() tensor.Tensor
	// GetUnits returns the number of units in the layer.
	GetUnits() int
	// GetUseBias returns true if the layer uses a bias.
	GetUseBias() bool
	//InitWeights initializes the weights of the layer.
	InitWeights()
	//GetNextLayer returns the next layer.
	GetNextLayer() []Layer
	//GetBefore gets the values of the previous layer.
	GetBefore() []Layer
	// SetNextLayer sets the next layer.
	SetNextLayer(Layer)
	// SetBefore sets the previous layer.
	SetBefore(Layer)
	// GetOutput the output of the layer.
	GetOutput() tensor.Tensor
	// FeedForward the input through the layer.
	FeedForward(pool workerpool.WorkerPool)
}
