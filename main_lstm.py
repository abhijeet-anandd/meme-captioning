from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np
import os
import cv2
import tkinter as tk
from tkinter import filedialog

folder = 'D:/project_image_captioning/images'
images = []
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
        images.append(img)

root = tk.Tk()
root.withdraw()
path = filedialog.askopenfilename()
x = path.rfind('/')
img_name_temp = path[x+1:]
img_name = img_name_temp[:len(img_name_temp)-4]
img_name = img_name.replace('-', ' ')

tokenizer = Tokenizer()
def dataset_preparation(data):
    corpus = data.lower().split("\n")	
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len, total_words

def create_model(predictors, label, max_sequence_len, total_words):
	model = Sequential()
	model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
	model.add(LSTM(150, return_sequences = True))
	model.add(LSTM(100))
	model.add(Dense(total_words, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
	model.fit(predictors, label, epochs=100, verbose=1, callbacks=[earlystop])
	print (model.summary())
	return model

def generate_text(seed_text, next_words, max_sequence_len):
	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
		predicted = model.predict_classes(token_list, verbose=0)
		
		output_word = ""
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word
	return seed_text

data = open('lstm.csv').read()

X, Y, max_len, total_words = dataset_preparation(data)
model = create_model(X, Y, max_len, total_words)
inp = input("Give me a seed word:	")
size_of_sentence = int(input("How many words do want in the generated sentence:   "))
text = generate_text(inp, size_of_sentence , max_len)
print (text)
                       