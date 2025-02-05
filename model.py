import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical  # Import for one-hot encoding

max_length = 100
vocab_size = 10000
embedding_dim = 16
oov_tok = "<OOV>"
trunc_type='post'
padding_type='post'

# Load your training data
train_df = pd.read_csv('twitterSentiment/twitter_training.csv')
train_df.columns = ['Column1', 'Column2', 'Column3', 'Column4']

# Preprocessing training data
train_df['Column4'] = train_df['Column4'].apply(lambda x: str(x) if isinstance(x, str) else '')

# Initialize and fit Tokenizer on training data
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(train_df['Column4'])

# Convert training texts to sequences and pad
train_sequences = tokenizer.texts_to_sequences(train_df['Column4'])
X_train = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Encode target variable for training data
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['Column3'])
y_train = to_categorical(y_train)  # One-hot encoding

# Load your validation data
val_df = pd.read_csv('twitterSentiment/twitter_validation.csv')
val_df.columns = ['Column1', 'Column2', 'Column3', 'Column4']

# Preprocessing validation data
val_df['Column4'] = val_df['Column4'].apply(lambda x: str(x) if isinstance(x, str) else '')

# Convert validation texts to sequences using the same Tokenizer
val_sequences = tokenizer.texts_to_sequences(val_df['Column4'])
X_val = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Encode target variable for validation data
y_val = label_encoder.transform(val_df['Column3'])
y_val = to_categorical(y_val)  # One-hot encoding for validation data

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

print("X_train shape:", X_train)
print("y_train shape:", y_train)
print("X_val shape:", X_val)
print("y_val shape:", y_val)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)

# Fit the model
history = model.fit(X_train, y_train,  
                    epochs=30, 
                    batch_size=16, 
                    verbose=1,
                    validation_data=(X_val, y_val), 
                    callbacks=[checkpoint])

# Save the model
model.save('my_model.h5')

# Predictions
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print the classification report
print(classification_report(np.argmax(y_val, axis=1), y_pred_classes))  # Compare y_val directly to y_pred_classes

# Print the confusion matrix
print(confusion_matrix(np.argmax(y_val, axis=1), y_pred_classes))

# Plotting training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotting training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
