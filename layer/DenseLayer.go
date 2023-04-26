package layer

import (
	"go-neural-toolkit/activations"
	"go-neural-toolkit/tensor"
	"go-neural-toolkit/workerpool"
	"reflect"
	"strings"
	"sync"
)

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
	// Activation function of the layer.
	Activation func(any)
}

// Dense creates the Dense Layer.
func Dense(units int, previous Layer, useBias bool, activation string, name string) *DenseLayer {
	l := &DenseLayer{
		Units: units, UseBias: useBias, Name: name,
	}
	// if useBias is true, we add 1 to the number of units.
	if useBias {
		l.Units++
	}
	// Add the previous Layer as Before Layer in this Layer
	l.SetBefore(previous)
	// Create the weights of the layer.
	l.InitWeights()
	// Set the Output of the Layer
	l.Output = tensor.CreateTensor1D(l.Units)
	// Set the activation // TODO: cases not working for me - why? Not importable?
	l.SetActivation(strings.Title(activation))
	// Add this Layer as next Layer in the previous Layer
	previous.SetNextLayer(l)
	return l
}

// SetActivation sets the activation function of the layer.
func (d *DenseLayer) SetActivation(activation string) {
	// Using refelect to get the function from the activation package.
	o := reflect.ValueOf(&activations.Activation).MethodByName(activation)
	if o.IsValid() {
		d.Activation = o.Interface().(func(any))
	} else {
		d.Activation = nil
	}
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
func (d *DenseLayer) GetBefore() []Layer {
	return d.Before // Dense Layer can only have one previous Layer
}

// InitWeights initializes the weights of the layer.
func (d *DenseLayer) InitWeights() {
	d.Weights = tensor.CreateTensor2D([]int{d.Units, d.Before[0].GetUnits()})
	return
}

// GetUseBias returns true if the layer uses a bias.
func (d *DenseLayer) GetUseBias() bool {
	return d.UseBias
}

// GetWeights returns the weights of the layer.
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

// DotProduct function calculates the dot product of two vectors.
func (d *DenseLayer) DotProduct(f any) {
	// f[0] is the input tensor, f[1] is the weight tensor, f[2] is the output tensor
	sum := float64(0)
	for idx, v := range *(f.([]*[]float64))[0] {
		sum += v * (*(f.([]*[]float64))[1])[idx]
	}
	(*(f.([]*[]float64))[2])[0] = sum
}

// FeedForward the input through the layer.
func (d *DenseLayer) FeedForward(pool *workerpool.WorkerPool) {
	// Create the WaitGroup
	var wg *sync.WaitGroup
	// switch the type of the former output type - we are expecting fpr a tensor.Tensor1D with Data = []float64
	switch outputBefore := d.GetBefore()[0].GetOutput().(type) {
	case *tensor.Tensor1D:
		wg = &sync.WaitGroup{}
		for idx, v := range d.Weights.(*tensor.Tensor2D).Data {
			wg.Add(1)
			pool.In <- workerpool.Workload{F: d.DotProduct, D: []*[]float64{&outputBefore.Data, &v, &[]float64{d.Output.(*tensor.Tensor1D).Data[idx]}}, Wg: wg}
		}
	default:
		// panic
		panic("Expecting a tensor.Tensor1D to layer " + d.GetName() + " but got " + reflect.TypeOf(outputBefore).String())
	}
	wg.Wait()
	// if useBias is true, we set the last value of the output to 1.
	if d.UseBias {
		d.Output.(*tensor.Tensor1D).Data[d.Units-1] = 1
	}
	// Now activate the output
	// TODO: Add logic for using diffrent activation functions
	for idx, v := range d.Output.(*tensor.Tensor1D).Data {
		wg.Add(1)
		pool.In <- workerpool.Workload{F: d.Activation, D: []*float64{&v, &d.Output.(*tensor.Tensor1D).Data[idx]}, Wg: wg}
	}
	wg.Wait()
}
