import json
import pyarabic.araby as araby
import tensorflow as tf
import numpy as np
import os
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from time import sleep


num_threads = 8
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
tf.config.set_soft_device_placement(True)
config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 16
config.inter_op_parallelism_threads = 16

tf.compat.v1.Session(config=config)
# tf.compat.v1.disable_v2_behavior()



# Load the tweet data
with open('final_for_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

tweets = [x[0] for x in data[0:500]]
replies = [x[1] for x in data[0:500]]

print(len(replies))
print(len(tweets))

tweets_tokens = [araby.tokenize(tweet) for tweet in tweets]
replies_tokens = [araby.tokenize(reply) for reply in replies]


vocab = set(token for tokens in tweets_tokens + replies_tokens for token in tokens)

# Create a mapping from token to index
token_to_index = {token: index for index, token in enumerate(vocab)}

# Convert the tweets_tokens and replies_tokens lists into a list of one-hot encoded vectors
tweets_dicts = [{token: 1 for token in tokens} for tokens in tweets_tokens]
replies_dicts = [{token: 1 for token in tokens} for tokens in replies_tokens]

vec = DictVectorizer()
tweets_encoded = vec.fit_transform(tweets_dicts).toarray()
replies_encoded = vec.transform(replies_dicts).toarray()
# Get the number of unique tokens
vocab_size = len(vocab)

# Pad the tweets and replies to the same length
max_length = max(len(tweet) for tweet in tweets_encoded)


print(f'{max_length = }')
print(f'{vocab_size = }')

tweets_padded = pad_sequences(tweets_encoded, maxlen=max_length, padding='post')
replies_padded = pad_sequences(replies_encoded, maxlen=max_length, padding='post')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(tweets_padded, replies_padded)
y_train = np.argmax(y_train, axis=1)
y_val = np.argmax(y_val, axis=1)
print(X_train[0:2])
print(tweets_padded[0:2])

print(len(X_train))
print(X_train.shape)
print(len(y_train))
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

del tweets_dicts, tweets_encoded, tweets_padded, tweets_tokens, tweets
del replies_dicts, replies_encoded, replies_padded, replies_tokens, replies


# with tf.device('/cpu:0'):
    # Define the model architecture
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length))
model.add(tf.keras.layers.LSTM(128))
model.add(tf.keras.layers.Dense(vocab_size, activation='sigmoid'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model

model.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_val, y_val))

# Save the trained model
model.save('model.h5')

# quit(1)




####################




# WARNING:tensorflow:Model was constructed with shape (None, 7655) for input KerasTensor(type_spec=TensorSpec(shape=(None, 7655), dtype=tf.float32, name='embedding_input'), name='embedding_input', description="created by layer 'embedding_input'"), but it was called on an input with incompatible shape (None, 7655, 7655).

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Tokenize the input tweet
input_tweet = "This is a sample tweet"
input_tokens = araby.tokenize(input_tweet)

# Convert the input tweet into a one-hot encoded vector
input_dict = {token: 1 for token in input_tokens}
input_vector = vec.transform(input_dict).toarray()

# Pad the input vector to the same length as the padded tweets used during training
input_vector_padded = pad_sequences([input_vector], maxlen=max_length, padding='post')

# Use the model to generate a reply
prediction = model.predict(input_vector_padded)

# Convert the one-hot encoded prediction back into tokens
prediction_tokens = vec.inverse_transform(prediction)

# Join the tokens into a single string
prediction_string = " ".join(prediction_tokens)

print(prediction_string)




# In this example, the tweet data is pre-processed by tokenizing it using the spaCy library, and then encoding it into numerical form using the Tokenizer class from the Keras API. The tweets and replies are then padded to the same length using the pad_sequences() function, and the data is split into training and validation sets using the train_test_split() function from scikit-learn.



# In the code example I provided, vocab_size is a variable that represents the size of the vocabulary used by the text AI model. The vocabulary is the set of unique words or tokens that the model is trained to recognize and generate.

# In the code example, vocab_size is used as an argument when defining the embedding layer of the model. The embedding layer is responsible for mapping the input text to a numerical representation that can be processed by the model. The vocab_size argument specifies the size of the vocabulary, and therefore determines the size of the embedding layer.

# To determine the value of vocab_size, you will need to count the number of unique tokens in the training data and set vocab_size equal to this value. For example, you could use the Tokenizer class from the Keras API to fit the training data and extract the vocabulary size as follows:

# Copy code
# # Pre-process the tweet data by tokenizing and encoding the text
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(tweets)
# vocab_size = len(tokenizer.word_index) + 1  # Add 1 for the padding token
# Alternatively, you can set vocab_size to a fixed value that is larger than the number of unique tokens in the training data. This can be useful if you want to leave room for additional tokens to be added to the vocabulary later, or if you want to use the same vocabulary size for multiple models. However, it is important to keep in mind that using a larger vocabulary size may require more computational resources and may not necessarily improve the model's performance.


