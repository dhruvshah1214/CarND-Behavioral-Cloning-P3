import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Lambda
from keras.regularizers import l2
import tensorflow as tf

def rgb2yuv(x):
    return tf.image.rgb_to_yuv(x)

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
	image = cv2.imread(full_path)
	image_rgb = image[...,::-1]
	images.append(image_rgb)
	measurement_steer = float(line[3])
	steer.append(measurement_steer)

X_train = np.array(images)
y_train = np.array(steer)

reg_constant = 1e-5
keep_prob = 0.5
activation_func = 'relu'

model = Sequential()
model.add(Lambda(rgb2yuv, input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: cv2.resize(x, (80, 160))))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, activation=activation_func, subsample=(2, 2), W_regulizer=l2(reg_constant)))
model.add(Convolution2D(36, 5, 5, activation=activation_func, subsample=(2, 2), W_regulizer=l2(reg_constant)))
model.add(Convolution2D(48, 5, 5, activation=activation_func, subsample=(2, 2), W_regulizer=l2(reg_constant)))
model.add(Convolution2D(64, 3, 3, activation=activation_func, W_regulizer=l2(reg_constant)))
model.add(Convolution2D(64, 3, 3, activation=activation_func, W_regulizer=l2(reg_constant)))
model.add(Flatten())
model.add(Dense(1164, activation=activation_func, W_regulizer=l2(reg_constant)))
model.add(Dense(100, activation=activation_func, W_regulizer=l2(reg_constant)))
model.add(Dense(50, activation=activation_func, W_regulizer=l2(reg_constant)))
model.add(Dense(10, activation=activation_func, W_regulizer=l2(reg_constant)))
model.add(Dense(1, activation=activation_func, W_regulizer=l2(reg_constant)))


model.compile(loss='mse', optimizer='adam')
print("FITTING")
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
