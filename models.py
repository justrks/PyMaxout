import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

def init_weights(irange, rng, shape):
    return rng.uniform(-irange, irange, shape)

class MLP:
    def set_layers(self, layers):
        self.layers = layers
        self.params = reduce(lambda l, layer: l + layer.params(), layers, [])
        self.velocities = map(lambda p: theano.shared(numpy.zeros_like(p.get_value())), self.params)

    def predict(self, x):
        return reduce(lambda z, layer: layer.predict(z), self.layers, x)

    def train(self, x):
        return reduce(lambda z, layer: layer.train(z), self.layers, x)

    def updates(self, cost, lr, momentum):
        grads = T.grad(cost, self.params)

        pairs = zip(self.params, self.velocities, grads)

        velocity_update = map(lambda (p, v, g): (v, lr * g + momentum * v), pairs)
        grads_update = map(lambda (p, v, g): (p, p - (lr * g + momentum * v)), pairs)

        return grads_update + velocity_update

class LinearRegressionLayer:
    def __init__(self, input_dim, irange, rng, name):
        self.w = theano.shared(rng.uniform(-irange, irange, (input_dim)), name=name + "_w")
        self.b = theano.shared(0.0, name=name + "_b")

    def predict(self, x):
        return T.dot(x, self.w) + self.b

    def train(self, x):
        return self.predict(x)

    def params(self):
        return [self.w, self.b]

class LogisticRegressionLayer:
    def __init__(self, input_dim, irange, name):
        self.w = theano.shared(rng.uniform(-irange, irange, (input_dim)), name=name + "_w")
        self.b = theano.shared(0, name=name + "_b")

    def predict(self, x):
        return 1 / (1 + T.exp(-T.dot(x, self.w) + self.b))

    def train(self, x):
        return self.predict(x)

    def params(self):
        return [self.w, self.b]

class MaxoutLayer:
    def __init__(self, input_dim, num_units, num_pieces, irange, dropout_rate, rng, name):
        self.w = theano.shared(init_weights(irange, rng, (num_units, input_dim, num_pieces)), name=name + "_w")
        self.b = theano.shared(init_weights(irange, rng, (num_units, num_pieces)), name=name + "_b")
        self.srng = RandomStreams(rng.randint(100))
        self.dropout_rate = dropout_rate

    def predict(self, x):
        k = T.dot(x, self.w) + self.b
        z = T.max(k, axis=2)
        return z

    def train(self, x):
        mask = self.srng.binomial(size=x.shape, p=self.dropout_rate, dtype=x.dtype)
        return self.predict(x * mask)

    def params(self):
        return [self.w, self.b]
