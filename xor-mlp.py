from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
# from keras.losses import mean_squared_error

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.metrics import mean_squared_error

# import seaborn as sns
# sns.set()

# C-c C-r f - fix indentation automatically

# base_X = [[0, 0, 1, 1],
#           [0, 1, 0, 1],
#           [-1, -1, -1, -1]]
base_X_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
base_t_data = [0, 1, 1, 0]


def generate_training_data(n):
    print "Generating training data..."
    noise = pd.DataFrame(np.random.normal(loc=0.0, scale=0.5, size=(n * 4, 2)))
    X = pd.DataFrame(base_X_data * n) + noise
    X["offset"] = [-1] * n * 4

    t = pd.DataFrame(base_t_data * n)
    X["t"] = t

    # print X

    return X


def display_accuracy_image(model, hidden_neurons, training_vectors):
    title = "{} hidden neurons, {} training vectors using hyperbolic tangent".format(hidden_neurons, training_vectors)
    filename = "images/image_tanh_{}neurons_{}vectors.png".format(hidden_neurons, training_vectors)

    splits = 30

    r = np.linspace(0, 1, splits, endpoint=True)

    df = pd.DataFrame(columns=["x", "y"])
    for y in r:
        df2 = pd.DataFrame()
        df2["x"] = r
        df2["y"] = y
        df = pd.concat([df, df2])
    df["offset"] = -1

    output = model.predict(df, batch_size=1)
    df["out"] = output

    del df["offset"]

    # print df
   
    del df["x"], df["y"]

    plt.figure()

    # ax = fig.add_axes([0, 1, 0, 1])
    # ax = fig.
    # ax.invert_yaxis()
    plt.imshow(
        df.values.reshape(splits, splits),
        cmap=cm.Greys_r,
        extent=[0, 1, 0, 1])

    plt.title(title)

    plt.savefig(filename)

def display_accuracy_epoch_graph(history, hidden_neurons, training_vectors):
    title = "{} hidden neurons, {} training vectors".format(hidden_neurons, training_vectors)
    filename = "images/loss_{}neurons_{}vectors.png".format(hidden_neurons, training_vectors)

    plt.figure()
    
    plt.plot(history.history["loss"])
    plt.title(title)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(filename)


def build_model(hidden_neurons):
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=3))
    model.add(Activation("tanh"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(optimizer=SGD(lr=0.1),
                  loss="mean_squared_error",
                  metrics=["acc"])

    return model


write_predictions = False


def run():
    f = None

    test_patterns = generate_training_data(64)
    ideal_predictions = test_patterns["t"]
    del test_patterns["t"]

    if write_predictions:
        f = open("prediction_accuracy.txt", "w")
    
    for data_size in [64]:
        for hidden_neurons in [4]:
            title = "{} hidden neurons, {} training vectors".format(hidden_neurons, data_size)
            print title

            model = build_model(hidden_neurons)
            # X = pd.read_csv("data_" + str(data_size) + "_good.csv")
            X = generate_training_data(data_size)

            t = X["t"]
            del X["t"]

            history = model.fit(X, t, epochs=500, batch_size=1)

            # display_accuracy_image(model, title)
            display_accuracy_image(model, hidden_neurons, data_size)
            # display_accuracy_epoch_graph(history, hidden_neurons, data_size)

            if write_predictions:

                predictions = model.predict(test_patterns, batch_size=1)

                error = mean_squared_error(ideal_predictions, predictions)
                print "Error: " + str(error)

                f.write(title + ": " + str(error) + "\n")


    plt.show()

    if write_predictions:
        f.close()

    # display_accuracy_epoch_graph(history)
    # model = build_model(8)
    # n = 16
    # X = generate_training_data(n)
    # X.to_csv("data_" + str(n) + ".csv", header=True, index=False)
    # X = pd.read_csv("data_64_good.csv")
    # print X
    # t = X["t"]
    # del X["t"]

    # history = model.fit(X, t, epochs=128, batch_size=1)

    # display_accuracy_image(model)
    # # display_accuracy_epoch_graph(history)


run()
