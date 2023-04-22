package tensor

// The type interface is the base interface for all tensors.
type Tensor interface {
	GetShape() any
	GetData() any
}

// Tensor1D is a 1D tensor.
type Tensor1D struct {
	// Shape of the Tensor
	Shape int
	// The data of the tensor.
	Data []float64
}

// CreateTensor1D creates a new 1D tensor.
func CreateTensor1D(shape int) *Tensor1D {
	return &Tensor1D{
		Data: make([]float64, shape), Shape: shape,
	}
}

// GetShape returns the shape of the tensor.
func (t *Tensor1D) GetShape() any {
	return t.Shape
}

// return the Data of the tensor.
func (t *Tensor1D) GetData() any {
	return t.Data
}

// Tensor2D is a 2D tensor.
type Tensor2D struct {
	// Shape of the Tensor
	Shape []int
	// The data of the tensor.
	Data [][]float64
}

// CreateTensor2D creates a new 2D tensor.
func CreateTensor2D(shape []int) *Tensor2D {
	tensor := &Tensor2D{
		Data: make([][]float64, shape[0]), Shape: shape,
	}
	for i := 0; i < shape[0]; i++ {
		tensor.Data[i] = make([]float64, shape[1])
	}
	return tensor
}

// GetShape returns the shape of the tensor.
func (t *Tensor2D) GetShape() any {
	return t.Shape
}

func (t *Tensor2D) GetData() any {
	return t.Data
}
