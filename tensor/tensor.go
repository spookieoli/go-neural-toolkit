package tensor

// Tensor1D is a 1D tensor.
type Tensor1D struct {
	// Shape of the Tensor
	Shape int
	// The data of the tensor.
	Data []float64
}

func (t *Tensor1D) CreateTensor1D(shape int) *Tensor1D {
	return &Tensor1D{
		Data: make([]float64, shape), Shape: shape,
	}
}

// Tensor2D is a 2D tensor.
type Tensor2D struct {
	// Shape of the Tensor
	Shape []int
	// The data of the tensor.
	Data [][]float64
}

func (t *Tensor2D) CreateTensor2D(shape []int) *Tensor2D {
	tensor := &Tensor2D{
		Data: make([][]float64, shape[0]), Shape: shape,
	}

	for i := 0; i < shape[0]; i++ {
		tensor.Data[i] = make([]float64, shape[1])
	}
	return tensor
}
