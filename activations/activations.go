package activations

import "math"

// empty struct
type Activations struct {
}

// Sigmoid function
func (l *Activations) Sigmoid(f *float64) float64 {
	// The sigmoid function is defined like the tensoflow sigmoid function
	return 1 / (1 + math.Exp(-*f))
}

// Tanh function
func (l *Activations) Tanh(f *float64) float64 {
	// The tanh function is defined like the tensoflow tanh function
	return math.Tanh(*f)
}

// Softmax function
func (l *Activations) Softmax(f *[]float64) []float64 {
	// The softmax function is defined like the tensoflow softmax function
	var sum float64
	ret := []float64{}
	for _, v := range *f {
		sum += math.Exp(v)
	}
	for _, v := range *f {
		ret = append(ret, math.Exp(v)/sum)
	}
	return ret
}

// Relu function
func (l *Activations) Relu(f *float64) float64 {
	// The relu function is defined like the tensoflow relu function
	if *f < 0 {
		return 0
	}
	return *f
}

// LeakyRelu function
func (l *Activations) LeakyRelu(f *float64) float64 {
	// The leaky relu function is defined like the tensoflow leaky relu function
	if *f < 0 {
		return 0.01 * *f
	}
	return *f
}

// Elu function
func (l *Activations) Elu(f *float64, alpha float64) float64 {
	// The elu function is defined like the tensoflow elu function
	if *f < 0 {
		return alpha * (math.Exp(*f) - 1)
	}
	return *f
}

// Selu function
func (l *Activations) Selu(f *float64, alpha float64, scale float64) float64 {
	// The selu function is defined like the tensoflow selu function
	if *f < 0 {
		return scale * alpha * (math.Exp(*f) - 1)
	}
	return scale * *f
}

// Softplus function
func (l *Activations) Softplus(f *float64) float64 {
	// The softplus function is defined like the tensoflow softplus function
	return math.Log(1 + math.Exp(*f))
}

// Softsign function
func (l *Activations) Softsign(f *float64) float64 {
	// The softsign function is defined like the tensoflow softsign function
	return *f / (1 + math.Abs(*f))
}

// HardSigmoid function
func (l *Activations) HardSigmoid(f *float64) float64 {
	// The hard sigmoid function is defined like the tensoflow hard sigmoid function
	if *f < -2.5 {
		return 0
	}
	if *f > 2.5 {
		return 1
	}
	return 0.2*(*f) + 0.5
}

// Exponential function
func (l *Activations) Exponential(f *float64) float64 {
	// The exponential function is defined like the tensoflow exponential function
	return math.Exp(*f)
}

// Linear function
func (l *Activations) Linear(f *float64) float64 {
	// The linear function is defined like the tensoflow linear function
	return *f
}

// ThresholdedRelu function
func (l *Activations) ThresholdedRelu(f *float64, theta float64) float64 {
	// The thresholded relu function is defined like the tensoflow thresholded relu function
	if *f < theta {
		return 0
	}
	return *f
}
