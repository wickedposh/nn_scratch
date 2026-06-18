Neural Network from Scratch

A multi-layer perceptron implemented in pure Numpy - forward pass, hand-derived backpropagation and momentum SGD, no autograd or ML frameworks.

I first wrote a 3-layer version in 2019 while self-teaching Python and auditing CMU's Deep Learning course(11-785). Returning to it recently, I reimplemented it froms scratch with a cleaner design: configurable depth (any number of hidden layers), three activation functions (sigmoid/tanh/ ReLU) and momentum-based gradient descent. Trains to convergence on XOR - a toy problem that famously stumped the early perceptrol.

