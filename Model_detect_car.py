
"""
Model for Car Recognition
"""

import numpy as np
import matplotlib.pyplot as plt

from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Convolution2D, Flatten, Input, Conv2D, MaxPooling2D, Lambda
from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split

def create_model(input_shape=(64,64,3)):
    filt = 64 # (64 x 64) size window for found car
    model = Sequential()
    model.add(Lambda(lambda x: x/(filt-0.5) - 1,input_shape=input_shape, output_shape=input_shape))
    model.add(Convolution2D(filt, (3, 3), activation='relu', name='conv1',input_shape=input_shape, border_mode="same"))  
    model.add(Dropout(0.5))
    model.add(Convolution2D(filt, (3, 3), activation='relu', name='conv2',border_mode="same"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(filt, (3, 3), activation='relu', name='conv3',border_mode="same"))
    model.add(MaxPooling2D(pool_size=(8,8)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(filt,(8,8),activation="relu",name="dense1")) 
    model.add(Dropout(0.5))
    model.add(Convolution2D(1,(1,1),name="dense2", activation="tanh"))
    model.summary()
    return model

"""
	Show result learning model
"""
def plot_results(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def train_model(model):
	data = np.load('venic.npy')
	X = np.array([img for img in data[:, 0]]).reshape(-1,64,64,3)
	Y = np.array([label for label in data[:, 1]])

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=63)

	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	model.add(Flatten())
	opt = optimizers.RMSprop(lr=0.0001)
	model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
	history = model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_test, Y_test))
	plot_results(history)

def load_model(model):
    model.load_weights('model_detect_car.h5')
    print("Weights loaded!")

def save_model(model):
    model.save_weights('model_detect_car.h5')
    print("Weights saved!")

def test_model(count=5):
	model = create_model()
	load_model(model)

	data = np.load('venic.npy')
	X = np.array([img for img in data[:, 0]]).reshape(-1,64,64,3)
	Y = np.array([label for label in data[:, 1]])

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=63)
	for i in range(count):
		rand = np.random.randint(X_test.shape[0])
		plt.imshow(X_test[rand])
		sample = np.reshape(X_test[rand], (1, 64,64,3))
		prediction = model.predict(sample, batch_size=64, verbose=0)
		prediction = prediction[0][0]
		
		if prediction >= 0.5:
			print("NN Prediction: CAR with value " + str(prediction))
		else:
			print("NN Prediction: NO CAR with value " + str(prediction))
		truth = Y_test[rand]
		if truth == 1:
			print("Ground-truth: CAR with value " + str(truth))
		else:
			print("Ground-truth: NO CAR with value " + str(truth))
		plt.show()

if __name__ == "__main__":
	model = create_model()
	train_model(model)
	save_model(model)
	#test_model()