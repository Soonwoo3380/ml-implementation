## Implementation Notes
- **Paper**: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017
- **Core Formula**: $Z = D^{-1/2} A D^{-1/2} X Θ$
- **Key Insight**: Simplifies spectral graph convolution via 1st-order Chebyshev approximation