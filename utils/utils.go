package utils

import (
	"go-neural-toolkit/tensor"
	"math"
	"math/rand"
)

// Empty struct
type Utils struct {
}

// Global utils variable
var Utilities Utils

// Init Utils
func init() {
	Utilities = Utils{}
}

// RandomNormal initializes the weights with the Random Normal method.
func (u *Utils) RandomNormal(w any) {
	for idx, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64) {
		for idy, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64)[idx] {
			(*(w.(*tensor.Tensor))).GetData().([][]float64)[idx][idy] = rand.NormFloat64()
		}
	}
}

// RandomUniform initializes the weights with the Random Uniform method.
func (u *Utils) RandomUniform(w any) {
	for idx, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64) {
		for idy, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64)[idx] {
			(*(w.(*tensor.Tensor))).GetData().([][]float64)[idx][idy] = rand.Float64()
		}
	}
}

// Zeros initializes the weights with the Zeros method.
func (u *Utils) Zeros(w any) {
	for idx, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64) {
		for idy, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64)[idx] {
			(*(w.(*tensor.Tensor))).GetData().([][]float64)[idx][idy] = 0
		}
	}
}

// Ones initializes the weights with the Ones method.
func (u *Utils) Ones(w any) {
	for idx, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64) {
		for idy, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64)[idx] {
			(*(w.(*tensor.Tensor))).GetData().([][]float64)[idx][idy] = 1
		}
	}
}

// TruncatedNormal initializes the weights with the Truncated Normal method.
func (u *Utils) TruncatedNormal(w any) {
	for idx, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64) {
		for idy, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64)[idx] {
			(*(w.(*tensor.Tensor))).GetData().([][]float64)[idx][idy] = rand.NormFloat64()
			for (*(w.(*tensor.Tensor))).GetData().([][]float64)[idx][idy] > 2 || (*(w.(*tensor.Tensor))).GetData().([][]float64)[idx][idy] < -2 {
				(*(w.(*tensor.Tensor))).GetData().([][]float64)[idx][idy] = rand.NormFloat64()
			}
		}
	}
}

// GlorotNormal initializes the weights with the Glorot Normal method. Glorot et al., 2010.
func (u *Utils) GlorotNormal(w any) {
	for idx, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64) {
		for idy, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64)[idx] {
			(*(w.(*tensor.Tensor))).GetData().([][]float64)[idx][idy] = rand.NormFloat64() * math.Sqrt(2.0/float64((*(w.(*tensor.Tensor))).GetShape().([]int)[0]+(*(w.(*tensor.Tensor))).GetShape().([]int)[1]))
		}
	}
}

// GlorotUniform initializes the weights with the Glorot Uniform method. Glorot et al., 2010.
func (u *Utils) GlorotUniform(w any) {
	for idx, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64) {
		for idy, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64)[idx] {
			(*(w.(*tensor.Tensor))).GetData().([][]float64)[idx][idy] = rand.Float64() * math.Sqrt(2.0/float64((*(w.(*tensor.Tensor))).GetShape().([]int)[0]+(*(w.(*tensor.Tensor))).GetShape().([]int)[1]))
		}
	}
}

// HeNormal initializes the weights with the He Normal method. He et al., 2015.
func (u *Utils) HeNormal(w any) {
	for idx, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64) {
		for idy, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64)[idx] {
			(*(w.(*tensor.Tensor))).GetData().([][]float64)[idx][idy] = rand.NormFloat64() * math.Sqrt(2.0/float64((*(w.(*tensor.Tensor))).GetShape().([]int)[0]))
		}
	}
}

// HeUniform initializes the weights with the He Uniform method. He et al., 2015.
func (u *Utils) HeUniform(w any) {
	for idx, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64) {
		for idy, _ := range (*(w.(*tensor.Tensor))).GetData().([][]float64)[idx] {
			(*(w.(*tensor.Tensor))).GetData().([][]float64)[idx][idy] = rand.Float64() * math.Sqrt(2.0/float64((*(w.(*tensor.Tensor))).GetShape().([]int)[0]))
		}
	}
}
