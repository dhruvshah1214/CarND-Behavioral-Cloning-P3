import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Lambda, Dropout
from keras.regularizers import l2
import tensorflow as tf
from keras.preprocessing import image as keras_image

def preprocess(x):
	yuv = cv2.cvtColor(x, cv2.COLOR_RGB2YUV)
	resize = cv2.resize(yuv, (160, 80))
	return yuv



if __name__ == '__main__':
	lines = []
	with open('data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		next(reader, None)	
		for line in reader:
			lines.append(line)

	print(len(lines))
	images = []
	steer = []
	for line in lines:
		path = line[0]
		filename = path.split('/')[-1]
		#print(filename)
		full_path = 'data/IMG/' + filename
		#print(full_path)
		image = cv2.imread(full_path)

		image_rgb = image[...,::-1]
		measurement_steer = float(line[3])
		images.append(preprocess(image_rgb))
		steer.append(measurement_steer)

		if steer != 0:

			steer_correction = 0.2

			path_left = 'data/IMG/' + line[1].split('/')[-1]
			img_left = cv2.imread(path_left)
			img_left_rgb = img_left[...,::-1]

			images.append(preprocess(img_left_rgb))
			steer.append(measurement_steer + steer_correction)

			path_right = 'data/IMG/' + line[2].split('/')[-1]
			img_right = cv2.imread(path_right)
			img_right_rgb = img_right[..., ::-1]

			images.append(preprocess(img_right_rgb))
			steer.append(measurement_steer - steer_correction)


	for (img, steer_val) in zip(images, steer):
		if steer_val != 0:
			img_flip = np.fliplr(img)
			measurement_flip = -steer_val
			images.append(preprocess(img_flip))
			steer.append(measurement_flip)




	X_train = np.array(images)
	y_train = np.array(steer)

	reg_constant = 1e-5
	keep_prob = 1.0
	activation_func = 'elu'

	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Convolution2D(24, 5, 5, activation=activation_func, subsample=(2, 2)))
	model.add(Convolution2D(36, 5, 5, activation=activation_func, subsample=(2, 2)))
	model.add(Convolution2D(48, 5, 5, activation=activation_func, subsample=(2, 2)))
	model.add(Convolution2D(64, 3, 3, activation=activation_func))
	model.add(Convolution2D(64, 3, 3, activation=activation_func))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(1164, activation=activation_func))
	model.add(Dense(100, activation=activation_func))
	model.add(Dense(50, activation=activation_func))
	model.add(Dense(10, activation=activation_func))
	model.add(Dense(1, activation=activation_func))
	
	
	model.compile(loss='mse', optimizer='adam')
	print("FITTING")

	history = model.fit(X_train, y_train, validation_split=0.2, nb_epoch=3, batch_size=128)
	
	print("TRAIN EVAL")
	model.evaluate(X_train, y_train, verbose=1)

	print("HISTORY")
	print(history.history)
	model.save('model.h5')
