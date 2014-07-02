import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import models
import costs

theano.config.openmp=True

##################
# Configurations #
##################
DATASET_PATH="./resources/dataset_shuf.csv"
NUM_EPOCHS = 1000000
NUM_FEATS = 11
NUM_UNITS = 10
NUM_PIECES = 5
WEIGHT_COST = 0.001
INITIAL_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.999996
MOMENTUM = 0.7
INIT_RANGE = 0.05

def normalize_data(D):
    max = D.max(axis=0)
    min = D.min(axis=0)
    return (D - min) / (max - min)

def train_function(x, y, model, cost):
    momentum = theano.shared(MOMENTUM, name="momentum")
    lr = theano.shared(INITIAL_LEARNING_RATE, name="lr")
    train = theano.function(
        inputs=[x, y],
        outputs=cost,
        updates=nn.updates(cost, lr, momentum) + [(lr, lr * LEARNING_RATE_DECAY)]
    )

    return train

if __name__ == "__main__":
    rng = numpy.random
    srng = RandomStreams(seed=123)

    # Load dataset
    D = numpy.loadtxt(DATASET_PATH, delimiter=";")
    Dfeat = normalize_data(D[:, 0:11])
    Dlabel = D[:, 11]

    Dtrain = zip(numpy.split(Dfeat[0:1000], 10), numpy.split(Dlabel[0:1000], 10))
    Dvalidate = [Dfeat[1000:], Dlabel[1000:]]

    # Model
    nn = models.MLP()
    l1 = models.MaxoutLayer(NUM_FEATS, NUM_UNITS, NUM_PIECES, INIT_RANGE, 0.8, rng, "l1")
    l2 = models.MaxoutLayer(NUM_UNITS, NUM_UNITS, NUM_PIECES, INIT_RANGE, 0.5, rng, "l2")
    l3 = models.LinearRegressionLayer(NUM_UNITS, INIT_RANGE, rng, "l3")
    nn.set_layers([l1, l2, l3])

    # Objective function
    x = T.matrix("x")
    y = T.vector("y")
    y_hat = nn.train(x)
    mse = costs.mean_squared_error(y, y_hat)
    cost = mse + WEIGHT_COST * costs.l1_norm(l1.w) + WEIGHT_COST * costs.l1_norm(l2.w)

    train = train_function(x, y, nn, cost)
    validate = theano.function(inputs=[x, y],
                               outputs=[nn.predict(x),
                                        costs.mean_squared_error(y, nn.predict(x))])

    # Train
    for i in range(NUM_EPOCHS):
        train_err = 0.0
        for minibatch in Dtrain:
            train_err += train(minibatch[0], minibatch[1])
        train_err /= len(Dtrain)

        if (i % 1000) == 0:
            prediction, validate_err = validate(Dvalidate[0], Dvalidate[1])
            print (i, train_err, validate_err)
