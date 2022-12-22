import json
import pyarabic.araby as araby
import tensorflow as tf
import numpy as np
import os
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


# Load the trained model
model = tf.keras.models.load_model('model.h5')

with open('final_for_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

tweets = [x[0] for x in data[0:500]]
replies = [x[1] for x in data[0:500]]

tweets_tokens = [araby.tokenize(tweet) for tweet in tweets]
replies_tokens = [araby.tokenize(reply) for reply in replies]

vocab = set(token for tokens in tweets_tokens + replies_tokens for token in tokens)
tweets_dicts = [{token: 1 for token in tokens} for tokens in tweets_tokens]


vec = DictVectorizer()
vec.fit(tweets_dicts)

max_length = 7655

# Tokenize the input tweet
input_tweet = "السلام عليكم"
input_tokens = araby.tokenize(input_tweet)

# Convert the input tweet into a one-hot encoded vector
input_dict = {token: 1 for token in input_tokens}
input_vector = vec.transform(input_dict).toarray()

# Pad the input vector to the same length as the padded tweets used during training
input_vector_padded = pad_sequences([input_vector], maxlen=max_length, padding='post')
input_vector_padded = np.argmax(input_vector_padded, axis=1)

# Use the model to generate a reply
prediction = model.predict(input_vector_padded)


# Convert the one-hot encoded prediction back into tokens
print(prediction)
print(len(prediction))
print(len(prediction[0]))
prediction_index = np.argmax(prediction)

# Look up the corresponding token in the vocabulary
prediction_token = list(vocab)[prediction_index]

# Join the tokens into a single string
prediction_string = " ".join(prediction_token)

print(prediction_string)