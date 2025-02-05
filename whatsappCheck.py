import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download stop words (only needed once)
nltk.download("stopwords")

# Get English stop words
stop_words = set(stopwords.words("english"))
print(stop_words)
custom_stop_words = ['<Media', 'omitted>']


# Open and read the file with UTF-8 encoding for special characters like emojis
with open("WhatsApp Chat with Siblings .txt", "r", encoding="utf-8") as f:
    content = f.read()

# Regular expression pattern to match dates in YYYY/MM/DD HH:MM format
pattern = r"\b\d{4}/\d{2}/\d{2}, \d{2}:\d{2} - \b"

# Split the content at each occurrence of the date pattern
split_content = re.split(pattern, content)

count = 0

words = []
messages = []
all_cleaned = ''

wordsCount = {}

# Print messages containing '👍' without sender names
for message in split_content:
    cleaned_message = re.sub(r"^[^:]+: ", "", message)
    messages.append(cleaned_message)
    for word in cleaned_message.split():
            words.append(word)
            if word.lower() not in stop_words and word.lower() not in custom_stop_words and isinstance(word, str):
                 all_cleaned = all_cleaned + word.lower() + ' '
    if '👍' in message:
        # Remove sender names (text before the first colon)
        print(cleaned_message, '\n')
        count += 1

for word in words:
    if word not in stop_words:
        if word.lower() in wordsCount:
            wordsCount[word.lower()] += 1
        else:
            wordsCount[word.lower()] = 1

sorted_data = dict(sorted(wordsCount.items(), key=lambda item: item[1]))
print(sorted_data)
print(all_cleaned)
print(words[0])
print("Count of messages with '👍':", count)

all_words = all_cleaned.split()

# Split all_words into chunks of 100 words
chunks = [" ".join(all_words[i:i + 100]) for i in range(0, len(all_words), 100)]

# Create a DataFrame from the chunks
df = pd.DataFrame(chunks, columns=["text"])

# Display the first few rows of the DataFrame to check the structure
print(df.head())

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

model = load_model('my_model.h5')

sequences = tokenizer.texts_to_sequences(df['text'])
X_predict = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

res = model.predict(X_predict)

predicted_classes = np.argmax(res, axis=1)

print(predicted_classes)

avg = 0
div = 0
for i in predicted_classes:
     if i > 0:
          avg += i
          div += 1

#0 is irrelevant, 1 negative, 2, neutral, 3 positive

print(avg / div)