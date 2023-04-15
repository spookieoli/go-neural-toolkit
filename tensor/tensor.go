package tensor

// The type interface is the base interface for all tensors.
type Tensor interface {
	GetShape(int) int
}

// Tensor1D is a 1D tensor.
type Tensor1D struct {
	// Shape of the Tensor
	Shape int
	// The data of the tensor.
	Data []float64
}

func CreateTensor1D(shape int) *Tensor1D {
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

func CreateTensor2D(shape []int) *Tensor2D {
	tensor := &Tensor2D{
		Data: make([][]float64, shape[0]), Shape: shape,
	}

	for i := 0; i < shape[0]; i++ {
		tensor.Data[i] = make([]float64, shape[1])
	}
	return tensor
}

func (t *Tensor2D) GetShape(i int) int {
	return t.Shape[i]
}
