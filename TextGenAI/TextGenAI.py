import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split  # Import train_test_split

data = ["It is a dark time for the Rebellion. "
        "Although the Death Star has been destroyed, "
        "Imperial troops have driven the Rebel forces from their hidden base and pursued them across the galaxy.",
        "Evading the dreaded Imperial Starfleet, a group of freedom fighters led by Luke Skywalker have established a new secret base on the remote ice world of Hoth.",
        "The evil lord Darth Vader, obsessed with finding young Skywalker, has dispatched thousands of remote probes into the far reaches of space."]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(data)

input_sequences = []
for sequence in sequences:
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_length-1),
    tf.keras.layers.LSTM(150, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=120, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

seed_text = "It is a dark time for the Rebellion."
next_words = 25

temperature = 0.85  # Adjust the temperature parameter

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')

    predicted = model.predict(token_list)[0]

    # Adjust the predicted probabilities with temperature
    predicted_adjusted = np.log(predicted) / temperature
    predicted_adjusted_exp = np.exp(predicted_adjusted)
    predicted_adjusted_probs = predicted_adjusted_exp / np.sum(predicted_adjusted_exp)

    # Sample the next word based on adjusted probabilities
    predicted_word_index = np.random.choice(len(predicted_adjusted_probs), size=1, p=predicted_adjusted_probs)[0]

    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)
