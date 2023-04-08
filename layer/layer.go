package layer

// The Layer interface is the base interface for all layers.
type Layer interface {
	// Name returns the name of the layer.
	Name() string
	// Layer returns the layer itself
	Layer() Layer
	// GetWeights returns the weights of the layer.
	GetWeights() *[]float64
	// GetUnits returns the number of units in the layer.
	GetUnits() int
	// UseBias returns true if the layer uses a bias.
	UseBias() bool
}
