import keras
from keras import layers 
from keras import ops 
import os
import pandas as pd


DIRPATH = os.path.dirname(__file__)

loaded_hobby = pd.read_csv(DIRPATH+"/Data/HobbyID.csv")

def load_userdb():
    return pd.read_csv(DIRPATH+"/Data/User_Preferrences.csv")


def load_recommender_model(fallback=True):
    
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
            return cls(getUserCount(), getHobbyCount(), EMBEDDING_SIZE)
    
    models = [ keras for keras in os.listdir(DIRPATH) if keras.endswith(".keras") ]
    vers = []
    for str in models:
        vers.append(str.removesuffix(".keras").split("-")[-1])
    vers = [ int(str) for str in vers if str.isdigit() ]
    vers.sort()

    try:
        latest = vers.pop() if len(vers) > 0 else 0
        print(f"trying to load: {DIRPATH}/Model-{latest}.keras")
        return keras.saving.load_model(DIRPATH+f"/Model-{latest}.keras")
    except Exception as e:
        print(f"failed to load latest. falling back to: Model.keras")
        print(e)
        return keras.saving.load_model(DIRPATH+f"/Model.keras")

def getHobbyIds():
    return loaded_hobby["hobbyId"].unique().tolist()

def getUserCount():
    return load_userdb()["userId"].nunique()

def getHobbyCount():
    return load_userdb()["hobbyId"].nunique()

def getUserHobbies(id, tried=True):
    userdb = load_userdb()
    userHobbies = userdb.loc[userdb["userId"] == id]["hobbyId"].tolist()
    if not tried:
        return [ x for x in getHobbyIds() if x not in userHobbies ]
    return userHobbies

def getHobbiesFromIds(hobby_list):
    return loaded_hobby.loc[loaded_hobby['hobbyId'].isin(hobby_list)]

# array = []
# for userid in userdb["userId"].unique().tolist():
#     array.append(userdb.loc[userdb["userId"] == userid]["hobbyId"].nunique())

# print(array)