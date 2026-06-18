Neural Network from Scratch

A multi-layer perceptron implemented in pure Numpy - forward pass, hand-derived backpropagation
and momentum, SGD, no autograd or ML frameworks.

I first wrote a 3-layer version in 2019 while self-eaching Python and auditing CMU's Deep Learning 
course(11-785). Returning to it recently, I reimplemented it froms cratch with a cleaner design:
configurable deption (any number of hidden layers), three activation functions (sigmoid/tanh/ ReLU)
and momemtum-based gradient descent. Trains to convergence on XOR.

