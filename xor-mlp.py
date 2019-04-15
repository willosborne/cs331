# CS331 Neural Computing coursework
# Will Osborne - u1603746
# willfosborne@gmail.com


# all this stuff needs to be installed:
# numpy
# pandas
# sklearn
# tensorflow
# keras
# install it with pip install x, or python -m pip install if that doesn't work
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.metrics import mean_squared_error

base_X_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
base_t_data = [0, 1, 1, 0]

def generate_training_data(n):
    print "Generatic training data..."
    # dataframe of gaussian noise
    noise = pd.DataFrame(np.random.normal(loc=0.0, scale=0.5, size=(n * 4, 2)))
    # add to base data and duplicate
    X = pd.DataFrame(base_X_data * n) + noise
    # add offset
    X["offset"] = [-1] * n * 4

    t = pd.DataFrame(base_t_data * n)
    X["t"] = t

    # fix columns
    X.columns = ["x", "y", "offset", "t"]

    return X


def display_accuracy_image(model, hidden_neurons, training_vectors):
    """Take a model and control parameters, and display the mapping as an image."""
    title = "{} hidden neurons, {} training vectors".format(hidden_neurons, training_vectors)
    filename = "images/image_{}neurons_{}vectors.png".format(hidden_neurons, training_vectors)

    splits = 30


    # generate an evenly-spaced 2D grid
    r = np.linspace(0, 1, splits, endpoint=True)

    # make a DataFrame of it for testing
    df = pd.DataFrame(columns=["x", "y"])
    for y in r:
        df2 = pd.DataFrame()
        df2["x"] = r
        df2["y"] = y
        df = pd.concat([df, df2])
    df["offset"] = -1

    # predict each x,y pair
    output = model.predict(np.array(df), batch_size=1)
    df["out"] = output

    del df["offset"]
    del df["x"], df["y"]

    plt.figure()

    # dump the grey values into an image and display it
    plt.imshow(
        df.values.reshape(splits, splits),
        cmap=cm.Greys_r,
        extent=[0, 1, 0, 1])

    plt.title(title)

    plt.savefig(filename)

def display_accuracy_epoch_graph(history, hidden_neurons, training_vectors):
    """Take a model and control data, and display the loss/epoch data from the fitting of that model"""
    title = "{} hidden neurons, {} training vectors".format(hidden_neurons, training_vectors)
    filename = "images/loss_{}neurons_{}vectors.png".format(hidden_neurons, training_vectors)

    plt.figure()
    
    # just plot it, nothing fancy here
    plt.plot(history.history["loss"])
    plt.title(title)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(filename)


def build_model(hidden_neurons):
    # use a Sequential model - nothing fancy
    model = Sequential()
    # input dimension is 3 to account for the offset column
    model.add(Dense(hidden_neurons, input_dim=3))
    model.add(Activation("sigmoid")) # activation func for first layer
    model.add(Dense(1)) # output layer
    model.add(Activation("sigmoid")) # ...and its activation func

    # compile with SGD, mean sq error and accuracy
    model.compile(optimizer=SGD(lr=0.1),
                  loss="mean_squared_error",
                  metrics=["acc"])

    return model

# can turn off the dumping of accuracies here
write_predictions = True


def run():
    f = None

    # make test pattern
    test_patterns = generate_training_data(64)
    ideal_predictions = test_patterns["t"]
    del test_patterns["t"]

    if write_predictions:
        f = open("prediction_accuracy.txt", "w")
    
    # for each combo of size and neuron count
    for data_size in [16, 32, 64]:
        for hidden_neurons in [2, 4, 8]:
            title = "{} hidden neurons, {} training vectors".format(hidden_neurons, data_size)
            print title

            # make a model as appropriate and data as appropriate
            model = build_model(hidden_neurons)
            X = generate_training_data(data_size)
            # alternatively load pre-generated data
            # X = pd.read_csv("data_" + str(data_size) + "_good.csv")

            # extract target
            t = X["t"]
            del X["t"]
		
            # fit the model to the data
            # np.array stuff is for compatibility w/ dcs versions of stuff
            # batch_size must be 1 or it behaves very differently - one vector at a time, please
            # 1000 epochs
            history = model.fit(np.array(X), np.array(t), nb_epoch=1000, batch_size=1)

            display_accuracy_image(model, hidden_neurons, data_size)
            display_accuracy_epoch_graph(history, hidden_neurons, data_size)

            if write_predictions:
                # dump predictions
                predictions = model.predict(np.array(test_patterns), batch_size=1)

                error = mean_squared_error(ideal_predictions, predictions)
                print "Error: " + str(error)

                f.write(title + ": " + str(error) + "\n")


    plt.show()

    if write_predictions:
        f.close()

run()

# that's all, folks!
# this was a fun coursework! many thanks :D
