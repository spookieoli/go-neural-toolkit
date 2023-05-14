package layer

import (
	"fmt"
	"go-neural-toolkit/activations"
	"go-neural-toolkit/tensor"
	"go-neural-toolkit/utils"
	"go-neural-toolkit/workerpool"
	"os"
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
	// Weights initialization method.
	WeightInitMethod string
	// Weights initialization Function
	WeightInitFunc func(any)
	// The Derivative of the Activation Function
	ActivationDerivative func(any)
	// The ErrorTensor of the Layer
	ErrorTensor *tensor.Tensor2D
	// IsOutputLayer returns true if the layer is an output layer.
	IsOutputLayer bool
}

// Dense creates the Dense Layer.
func Dense(units int, previous Layer, useBias bool, activation string, name string, weightInitMethod string) *DenseLayer {
	l := &DenseLayer{
		Units: units, UseBias: useBias, Name: name,
	}
	l.WeightInitMethod = weightInitMethod
	if l.WeightInitMethod == "" {
		fmt.Println("No WeightInitMethod given - Init weights with random values")
	} else {
		l.SetWeightInitMethod(l.WeightInitMethod)
	}
	// Add the previous Layer as Before Layer in this Layer
	l.SetBefore(previous)
	// Create the weights of the layer.
	l.InitWeights()
	// Set the Output of the Layer
	l.Output = tensor.CreateTensor1D(l.Units)
	// Every neuron has its own ErrorTensor - the Errortensor is a Tensor2d
	l.ErrorTensor = tensor.CreateTensor2D([]int{0, 0})
	l.ErrorTensor.Data = make([][]float64, 0)
	// Set the activation // TODO: cases not working for me - why? Not importable?
	l.SetActivation(strings.Title(activation))
	// Add this Layer as next Layer in the previous Layer
	previous.SetNextLayer(l)
	return l
}

// SetWeightInitMethod sets the weights initialization method.
func (d *DenseLayer) SetWeightInitMethod(method string) {
	// Using reflect to get the function from the activation package.
	o := reflect.ValueOf(&utils.Utilities).MethodByName(method)
	if o.IsValid() {
		d.WeightInitFunc = o.Interface().(func(any))
	} else {
		d.WeightInitFunc = nil
	}
}

// SetActivation sets the activation function of the layer.
func (d *DenseLayer) SetActivation(activation string) {
	// Using refelect to get the function from the activation package.
	o := reflect.ValueOf(&activations.Activation).MethodByName(activation)
	if o.IsValid() {
		d.Activation = o.Interface().(func(any))
	} else {
		d.Activation = activations.Activation.Linear
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
	if d.UseBias {
		d.Weights = tensor.CreateTensor2D([]int{d.Units, d.Before[0].GetUnits() + 1})
	} else {
		d.Weights = tensor.CreateTensor2D([]int{d.Units, d.Before[0].GetUnits()})
	}
	// Now initialize the weights
	if d.WeightInitMethod != "" {
		d.WeightInitFunc(&d.Weights)
	} else {
		// If no WeightInitMethod is given, use pseudo random values
		utils.Utilities.RandomUniform(&d.Weights)
	}
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

// GetErrorTensor returns the error tensor of the layer.
func (d *DenseLayer) GetErrorTensor() *tensor.Tensor2D {
	return d.ErrorTensor
}

// DotProduct function calculates the dot product of two vectors.
func (d *DenseLayer) DotProduct(f any) {
	// f[0] is the input tensor, f[1] is the weight tensor, f[2] is the output tensor
	sum := float64(0)
	for idx, v := range *(f.([]any)[0].(*[]float64)) {
		sum += v * (*(f.([]any)[1].(*[]float64)))[idx]
	}
	// If useBias is true, add the bias to the sum
	if d.UseBias {
		sum += 1.0 * (*(f.([]any)[1].(*[]float64)))[len(*(f.([]any)[1].(*[]float64)))-1]
	}
	*(f.([]any)[2].(*float64)) = sum
}

// IsOutput returns true if the layer is an output layer.
func (d *DenseLayer) IsOutput() bool {
	return len(d.NextLayer) == 0
}

// SetisOutput sets the layer as an output layer.
func (d *DenseLayer) SetIsOutput(b bool) {
	d.IsOutputLayer = b
}

// GetInput gets the data for the ErrorTensor
func (d *DenseLayer) GetErrorData() *tensor.Tensor2D {
	return d.ErrorTensor
}

// FeedForward the input through the layer.
func (d *DenseLayer) FeedForward(pool *workerpool.WorkerPool) {
	// Create the WaitGroup
	var wg *sync.WaitGroup

	// switch the type of the former output type - we are expecting fpr a tensor.Tensor1D with Data = []float64
	switch outputBefore := d.GetBefore()[0].GetOutput().(type) {
	case *tensor.Tensor1D:
		wg = &sync.WaitGroup{}
		for idx, _ := range d.Output.(*tensor.Tensor1D).Data {
			wg.Add(1)
			pool.In <- workerpool.Workload{F: d.DotProduct, D: []any{&outputBefore.Data, &d.Weights.(*tensor.Tensor2D).Data[idx], &d.Output.(*tensor.Tensor1D).Data[idx]}, Wg: wg}
		}
		wg.Wait()

		// Now activate the output
		for _, v := range d.Output.(*tensor.Tensor1D).Data {
			wg.Add(1)
			pool.In <- workerpool.Workload{F: d.Activation, D: []*float64{&v}, Wg: wg}
		}
		wg.Wait()
		// Add Data to the error tensor for every neuron and every input
		for i := 0; i < d.Units; i++ {
			for j := 0; j < len(d.Before[0].GetOutput().GetData().([]float64)); j++ {
				d.ErrorTensor.Data = append(d.ErrorTensor.Data, []float64{d.Before[0].GetOutput().GetData().([]float64)[j], 2.0, 3.0, 4.0})
			}
		}
		fmt.Println(d.ErrorTensor.Data)
		os.Exit(0)
	default:
		// panic
		panic("Expecting a tensor.Tensor1D to layer " + d.GetName() + " but got " + reflect.TypeOf(outputBefore).String())
	}
}
