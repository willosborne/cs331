from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.losses import mean_squared_error

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
sns.set()

# C-c C-r f - fix indentation automatically

# base_X = [[0, 0, 1, 1],
#           [0, 1, 0, 1],
#           [-1, -1, -1, -1]]
base_X_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
base_t_data = [0, 1, 1, 0]




def generate_training_data(n):
    print "Generatic training data..."
    noise = pd.DataFrame(np.random.normal(loc=0.0, scale=0.5, size=(n * 4, 2)))
    # print noise
    X = pd.DataFrame(base_X_data * n) + noise
    X["offset"] = [-1] * n * 4

    t = pd.DataFrame(base_t_data * n)
    X["t"] = t

    # print X

    return X


def run():

    model = Sequential()
    model.add(Dense(16, input_dim=3))
    model.add(Activation("sigmoid"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(
        optimizer=SGD(lr=0.1), loss="mean_squared_error", metrics=["accuracy"])

    X = generate_training_data(64)
    t = X["t"]
    del X["t"]

    model.fit(X, t, epochs=100, batch_size=1)

    # print model.predict_proba(X)

    # fig = plt.figure()

    g = pd.DataFrame()

    splits = 30
    
    r = np.linspace(0, 1, splits, endpoint=True)

    df = pd.DataFrame(columns=["x", "y"])
    for y in r:
        df2 = pd.DataFrame()
        df2["x"] = r
        df2["y"] = y
        df = pd.concat([df, df2])
    df["offset"] = -1

    output = model.predict(df)
    df["out"] = output

    del df["offset"]

    print df

    del df["x"], df["y"]
    plt.imshow(df.values.reshape(splits, splits), cmap=cm.Greys_r)

    plt.show()

    df = pd.DataFrame(base_X_data)
    df["offset"] = -1
    out = model.predict(df)
    df["out"] = out

    print df


run()
