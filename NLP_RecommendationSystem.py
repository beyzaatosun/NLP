#%% Oneri Sistemi - Deep Learning
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot,Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

user_ids = np.array([0,1,2,3,4,0,1,2,3,4])
item_ids = np.array([0,1,2,3,4,1,2,3,4,5])
ratings =  np.array([5,4,3,2,1,4,5,3,2,1])

user_ids_train, user_ids_test, item_ids_train, item_ids_test, ratings_train, ratings_test = train_test_split(user_ids, item_ids, ratings, test_size=0.2,random_state=42)

def create_model(num_users, num_items, embedding_dim):
    #input layer
    user_input = Input(shape=(1,), name="user")
    item_input = Input(shape=(1,), name="item")
    #embedding layer
    user_embedding = Embedding(input_dim= num_users, output_dim = embedding_dim, name="user_embedding")(user_input)
    item_embedding = Embedding(input_dim = num_items, output_dim = embedding_dim, name="item_embedding")(item_input)
    
    #vektor duzlestir
    user_vecs = Flatten()(user_embedding)
    item_vecs = Flatten()(item_embedding)
    
    dot_product = Dot(axes=1)([user_vecs, item_vecs])
    output = Dense(1)(dot_product)
    
    model = Model(inputs = [user_input, item_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate = 0.001), loss="mean_squared_error")
    
    return model


num_users = 5
num_items = 6
embedding_dim = 8
model = create_model(num_users, num_items, embedding_dim)
model.fit([user_ids_train, item_ids_train], ratings_train, epochs=100, verbose=1, validation_split=0.1)

loss = model.evaluate([user_ids_test, item_ids_test], ratings_test)

print(f"test: {loss}")

user_id = np.array([0])
item_id = np.array([5])


prediction = model.predict([user_id, item_id])

print(f"Predicted rating for user: {user_id[0]}, item: {item_id[0]}, {prediction[0][0]:.2f}")

#%% Oneri Sistemi - Machine Learning

from surprise import Dataset, KNNBasic, accuracy
from surprise.model_selection import train_test_split

data = Dataset.load_builtin("ml-100k")

trainset, testset = train_test_split(data,test_size=0.2)

model_options = {
    "name":"cosine",
    "user_based":True #kullanicilar arasi benzerlik
    }

model = KNNBasic(sim_options=model_options)
model.fit(trainset)

prediction = model.test(testset)
accuracy.rmse(prediction)


def get_top_n(predictions, n=10):
    top_n ={}
    for uid, iid, true_r, est, _ in predictions:
        if not top_n.get(uid):
            top_n[uid]=[]
        top_n[uid].append((iid,est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1],reverse=True)
    return top_n
n=5
top_n = get_top_n(prediction,n)

user_id ="2"
print(f"top {n} recommendation for user {user_id}")

for item_id, rating in top_n[user_id]:
    print(f"item id: {item_id}, score: {rating}")