import pandas as pd
from pathlib import Path
# import matplotlib.pyplot as plt
import numpy as np
from zipfile import ZipFile

import os
import keras
from keras import layers
from keras import ops

from load_model import load_userdb
DIRPATH = os.path.dirname(__file__)

def train_model(filename):
    df = load_userdb()
    df = df.dropna(how='any',axis=0)
    df.isnull().any()

    user_ids = df["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    hobby_ids = df["hobbyId"].unique().tolist()
    hobby2hobby_encoded = {x: i for i, x in enumerate(hobby_ids)}
    hobby_encoded2hobby = {i: x for i, x in enumerate(hobby_ids)}
    df["user"] = df["userId"].map(user2user_encoded)
    df["hobby"] = df["hobbyId"].map(hobby2hobby_encoded)

    num_users = len(user2user_encoded)
    num_hobbies = len(hobby_encoded2hobby)
    df["rating"] = df["rating"].values.astype(np.float32)
    # min and max ratings will be used to normalize the ratings later
    min_rating = min(df["rating"])
    max_rating = max(df["rating"])

    print(
        "Number of users: {}, Number of Hobbies: {}, Min rating: {}, Max rating: {}".format(
            num_users, num_hobbies, min_rating, max_rating
        )
    )


    df = df.sample(frac=1, random_state=42)
    x = df[["user", "hobby"]].values
    # Normalize the targets between 0 and 1. Makes it easy to train.
    y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    # Assuming training on 90% of the data and validating on 10%.
    train_indices = int(0.8 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:],
    )
    len(x_train)

    EMBEDDING_SIZE = 50

    @keras.saving.register_keras_serializable()
    class RecommenderNet(keras.Model):
        def __init__(self, num_users, num_hobby, embedding_size, **kwargs):
            super().__init__(**kwargs)
            self.num_users = num_users
            self.num_hobby = num_hobby
            self.embedding_size = embedding_size
            self.user_embedding = layers.Embedding(
                num_users,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
            )
            self.user_bias = layers.Embedding(num_users, 1)
            self.hobby_embedding = layers.Embedding(
                num_hobby,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
            )
            self.hobby_bias = layers.Embedding(num_hobby, 1)

        def call(self, inputs):
            user_vector = self.user_embedding(inputs[:, 0])
            user_bias = self.user_bias(inputs[:, 0])
            hobby_vector = self.hobby_embedding(inputs[:, 1])
            hobby_bias = self.hobby_bias(inputs[:, 1])
            dot_user_hobby = ops.tensordot(user_vector, hobby_vector, 2)
            # Add all the components (including bias)
            x = dot_user_hobby + user_bias + hobby_bias
            # The sigmoid activation forces the rating to between 0 and 1
            return ops.nn.sigmoid(x)

        @classmethod
        def from_config(cls, self):
            return cls(num_users, num_hobbies, EMBEDDING_SIZE)
    
    model = RecommenderNet(num_users, num_hobbies, EMBEDDING_SIZE)
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=5,
        verbose=1,
        validation_data=(x_val, y_val),
    )

    model.save(DIRPATH+"/"+filename)

def start():
    models = [ keras for keras in os.listdir(DIRPATH) if keras.endswith(".keras") ]
    vers = []
    for str in models:
        vers.append(str.removesuffix(".keras").split("-")[-1])
    print(vers)
    vers = [ int(str) for str in vers if str.isdigit() ]

    vers.sort()
    try:
        latest = vers.pop() if len(vers) > 0 else 0
        train_model(f"Model-{latest+1}.keras")
    except:
        print("Could not train model")
start()