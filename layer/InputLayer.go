package layer

import "go-neural-toolkit/tensor"

type InputLayer struct {
	// The name of the layer.
	Name string
	// The number of units in the layer.
	Units int
	// The weights of the layer.
	Weights tensor.Tensor
	// The output of the layer.
	Output tensor.Tensor
	// The next Layer
	NextLayer []Layer
	// The previous Layer
	Before []Layer
	// If the layer uses a bias.
	UseBias bool
	// Activation function of the layer.
	Activation func(any)
}

// Input creates the Input Layer.
// *NOTE* input is the number auf Data Columns in the Dataset given.
func Input(units int, name string, shape []int) *InputLayer {
	l := &InputLayer{
		Units: units, Name: name,
	}

	// Create the outputs - the input layer is just a placeholder for the input data.
	if len(shape) == 1 {
		l.Output = tensor.CreateTensor1D(l.Units)
	} else {
		l.Output = tensor.CreateTensor2D(shape)
	}
	return l
}

// GetName returns the name of the layer.
func (i *InputLayer) GetName() string {
	return i.Name
}

// GetWeights returns the weights of the layer.
func (i *InputLayer) GetWeights() tensor.Tensor {
	return nil
}

// GetUnits returns the number of units in the layer.
func (i *InputLayer) GetUnits() int {
	return i.Units
}

// GetUseBias returns true if the layer uses a bias.
func (i *InputLayer) GetUseBias() bool {
	return false
}

// InitWeights initializes the weights of the layer.
func (i *InputLayer) InitWeights() {
	return
}

// GetNextLayer returns the next layer.
func (i *InputLayer) GetNextLayer() []Layer {
	return i.NextLayer
}

// GetBefore gets the values of the previous layer.
func (i *InputLayer) GetBefore() []Layer {
	return nil // Input Layer has no previous Layer
}

// SetNextLayer sets the next layer.
func (i *InputLayer) SetNextLayer(l Layer) {
	i.NextLayer = append(i.NextLayer, l)
}

// SetBefore sets the previous layer.
func (i *InputLayer) SetBefore(l Layer) {
	return // Input Layer has no previous Layer
}

// GetOutput the output of the layer.
func (i *InputLayer) GetOutput() tensor.Tensor {
	return i.Output
}

// FeedForward the input through the layer.
func (i *InputLayer) FeedForward() {
	return
}
