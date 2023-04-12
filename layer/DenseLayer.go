package layer

import "go-neural-toolkit/tensor"

type DenseLayer struct {
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
}

// Dense creates the Dense Layer.
func Dense(units int, previous Layer, useBias bool, activation string, name string) *DenseLayer {
	l := &DenseLayer{
		Units: units, UseBias: useBias, Name: name,
	}
	// Add this Layer as next Layer in the previous Layer
	previous.SetNextLayer(l)
	// Add the previous Layer as Before Layer in this Layer
	l.SetBefore(previous)
	return l
}

// SetNextLayer sets the next layer.
func (d *DenseLayer) SetNextLayer(l Layer) {
	d.NextLayer = append(d.NextLayer, l)
}

// SetBefore sets the previous layer.
func (d *DenseLayer) SetBefore(l Layer) {
	d.Before = append(d.Before, l)
}

// GetNextLayer returns the next layer.
func (d *DenseLayer) GetNextLayer() []Layer {
	return d.NextLayer
}

// GetBefore gets the values of the previous layer.
func (d *DenseLayer) GetBefore() Layer {
	return d.Before[0] // Dense Layer can only have one previous Layer
}

// initWeights initializes the weights of the layer.
func (d *DenseLayer) InitWeights() {
	// TODO IMPLEMENT
	return
}

// USeBias returns true if the layer uses a bias.
func (d *DenseLayer) GetUseBias() bool {
	return d.UseBias
}

// Getweights returns the weights of the layer.
func (d *DenseLayer) GetWeights() tensor.Tensor {
	return d.Weights
}

// GetName returns the name of the layer.
func (d *DenseLayer) GetName() string {
	return d.Name
}

// GetOutput returns the output of the layer.
func (d *DenseLayer) GetOutput() tensor.Tensor {
	return d.Output
}

// GetUnits returns the number of units in the layer.
func (d *DenseLayer) GetUnits() int {
	return d.Units
}
