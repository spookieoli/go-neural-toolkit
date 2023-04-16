package activations

import "math"

// empty struct
type Activations struct {
}

// global activation variable
var Activation Activations

// Init the activations
func init() {
	Activation = Activations{}
}

// Sigmoid function
func (l *Activations) Sigmoid(f any) {
	// The sigmoid function is defined like the tensoflow sigmoid function
	*(f.([]*float64))[1] = 1 / (1 + math.Exp(-*(f.([]*float64))[0]))
}

// Tanh function
func (l *Activations) Tanh(f any) {
	// The tanh function is defined like the tensoflow tanh function
	*(f.([]*float64))[1] = math.Tanh(*(f.([]*float64))[1])
}

// Softmax function
func (l *Activations) Softmax(f any) {
	// The softmax function is defined like the tensoflow softmax function
	var sum float64
	for _, v := range f.([]*float64) {
		sum += math.Exp(*v)
	}
	for _, v := range f.([]*float64) {
		*v = math.Exp(*v) / sum
	}
}

// Relu function
func (l *Activations) Relu(f any) {
	// The relu function is defined like the tensoflow relu function
	if *(f.([]*float64))[0] < 0 {
		*(f.([]*float64))[1] = 0
	}
	*(f.([]*float64))[1] = *(f.([]*float64))[0]
}

// LeakyRelu function
func (l *Activations) LeakyRelu(f any) {
	// The leaky relu function is defined like the tensoflow leaky relu function
	if *(f.([]*float64))[0] < 0 {
		*(f.([]*float64))[1] = 0.01 * *(f.([]*float64))[0]
	}
	*(f.([]*float64))[1] = *(f.([]*float64))[0]
}

// Elu function
func (l *Activations) Elu(f any) {
	// The elu function is defined like the tensoflow elu function
	if *(f.([]*float64))[0] < 0 {
		*(f.([]*float64))[2] = *(f.([]*float64))[1] * (math.Exp(*(f.([]*float64))[0]) - 1)
	}
	*(f.([]*float64))[2] = *(f.([]*float64))[0]
}

// Selu function
func (l *Activations) Selu(f any) {
	// The selu function is defined like the tensoflow selu function - scale is f[1], alpha is f[2], result is f[3]
	if *(f.([]*float64))[0] < 0 {
		*(f.([]*float64))[3] = *(f.([]*float64))[1] * *(f.([]*float64))[2] * (math.Exp(*(f.([]*float64))[0]) - 1)
	}
	*(f.([]*float64))[3] = *(f.([]*float64))[1] * *(f.([]*float64))[0]
}

// Softplus function
func (l *Activations) Softplus(f any) {
	// The softplus function is defined like the tensoflow softplus function
	*(f.([]*float64))[1] = math.Log(1 + math.Exp(*(f.([]*float64))[0]))
}

// Softsign function
func (l *Activations) Softsign(f any) {
	// The softsign function is defined like the tensoflow softsign function
	*(f.([]*float64))[1] = *(f.([]*float64))[0] / (1 + math.Abs(*(f.([]*float64))[0]))
}

// HardSigmoid function
func (l *Activations) HardSigmoid(f any) {
	// The hard sigmoid function is defined like the tensoflow hard sigmoid function
	if *(f.([]*float64))[0] < -2.5 {
		*(f.([]*float64))[1] = 0
	}
	if *(f.([]*float64))[0] > 2.5 {
		*(f.([]*float64))[1] = 1
	}
	*(f.([]*float64))[1] = 0.2*(*(f.([]*float64))[0]) + 0.5
}

// Exponential function
func (l *Activations) Exponential(f any) {
	// The exponential function is defined like the tensoflow exponential function
	*(f.([]*float64))[1] = math.Exp(*(f.([]*float64))[0])
}

// Linear function
func (l *Activations) Linear(f any) {
	// The linear function is defined like the tensoflow linear function
	*(f.([]*float64))[1] = *(f.([]*float64))[0]
}

// ThresholdedRelu function
func (l *Activations) ThresholdedRelu(f any) {
	// The thresholded relu function is defined like the tensoflow thresholded relu function
	if *(f.([]*float64))[0] < *(f.([]*float64))[1] {
		*(f.([]*float64))[2] = 0
	}
	*(f.([]*float64))[2] = *(f.([]*float64))[0]
}
