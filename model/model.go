package model

import (
	"fmt"
	"go-neural-toolkit/layer"
	"go-neural-toolkit/tensor"
	"go-neural-toolkit/workerpool"
)

type ModelConfig struct {
	// The InputLayers of the Model.
	InputLayers []layer.Layer
	// The OutputLayers of the Model
	OutputLayers []layer.Layer
}

type Model struct {
	// The layers of the model.
	Layers []layer.Layer
	// The WorkerPool of the Model.
	Workerpool *workerpool.WorkerPool
	// The LayerArray of the Model.
	LayerArray []layer.Layer // This Array will be used to store the layers in the right order.
	Loss       tensor.Tensor
}

// NewModel creates a new model. NewModel takes only the Input Layers of the Model and the number of worker threads.
func NewModel(mc *ModelConfig, worker int) (*Model, error) {
	// Guard clause
	if len(mc.InputLayers) == 0 {
		return nil, fmt.Errorf("No Layers given")
	}
	// Check if all Layers given are Inputlayers
	if CheckIsInputLayer(mc.InputLayers) == false {
		return nil, fmt.Errorf("No Input Layer given")
	}
	// Create the Layer Array
	la := make([]layer.Layer, 0, 20)
	// Fill rest of the Layers in the Layer Array
	FillLayerArray(mc.InputLayers, &la)
	// Create the Model
	return &Model{
		Layers:     mc.InputLayers,
		Workerpool: workerpool.NewWorkerPool(worker),
		LayerArray: la,
	}, nil
}

// The Predict Method
func (m *Model) Predict(input []tensor.Tensor) []tensor.Tensor {
	// Guard Clause
	if len(input) != len(m.Layers) {
		fmt.Println("Total Number of Input Layers and Input Tensors do not match")
		return nil
	}
	// TODO: NOT FINISHED YET
	fmt.Println("Predict Method not finished yet")
	return nil
}

// FillLayerArray creates the layer array.
func FillLayerArray(layers []layer.Layer, la *[]layer.Layer) {
	for _, l := range layers {
		if (CheckIfLayerExists(l, *la) == false || l.GetBefore() != nil) && CheckIfArrExistsInArr(l.GetBefore(), *la) == true {
			*la = append(*la, l)
			FillLayerArray(l.GetNextLayer(), la)
		}
	}
}

// CheckIsInputLayer if the given layers are of Type InputLayer
func CheckIsInputLayer(layers []layer.Layer) bool {
	for _, l := range layers {
		if _, ok := l.(*layer.InputLayer); !ok {
			return false
		}
	}
	return true
}

// CheckIfLayerExists checks if the given layer exists in the layer array.
func CheckIfLayerExists(layer layer.Layer, la []layer.Layer) bool {
	for _, l := range la {
		if l == layer {
			return true
		}
	}
	return false
}

// The CheckIfArrExistsInArr checks if the Layers already exist in the Layer Array.
func CheckIfArrExistsInArr(layers []layer.Layer, la []layer.Layer) bool {
	ba := []bool{}
	for _, l := range layers {
		for _, l2 := range la {
			if l == l2 {
				ba = append(ba, true)
			}
		}
	}
	if len(ba) == len(layers) {
		return true
	}
	return false
}
