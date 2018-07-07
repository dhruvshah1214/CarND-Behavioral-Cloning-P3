import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import argparse

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Lambda, Dropout, Cropping2D
from keras.regularizers import l2
import tensorflow as tf
from keras.preprocessing import image as keras_image


def readAllData(root_paths):
	image_paths = []
	steer = []
	for root_path in root_paths:
		lines = []
		with open(root_path + 'driving_log.csv') as csvfile:
			reader = csv.reader(csvfile)
			next(reader, None)
			for line in reader:
				lines.append(line)

		for line in lines:

			# skip low speeds
			if float(line[6]) < 5.0:
				continue

			path = line[0]
			filename = path.split('/')[-1]
			# print(filename)
			full_path = root_path + 'IMG/' + filename
			# print(full_path)
			measurement_steer = float(line[3])

			image_paths.append(full_path)
			steer.append(measurement_steer)

			steer_correction = 0.25

			path_left = root_path + 'IMG/' + line[1].split('/')[-1]

			image_paths.append(path_left)
			steer.append(measurement_steer + steer_correction)

			path_right = root_path + 'IMG/' + line[2].split('/')[-1]

			image_paths.append(path_right)
			steer.append(measurement_steer - steer_correction)

	print("File reads: " + str(len(steer)))
	return (np.array(image_paths), np.array(steer))

def angleDistribution(angles):
	num_bins = 20
	avg_samples_per_bin = len(angles) / num_bins
	plt.hist(angles, bins=num_bins)
	plt.show()
	return  avg_samples_per_bin

def lowerZeroes(image_paths, steer, keep_prob=0.4):
	image_path_new = []
	steer_new = []
	for i in range(len(steer)):
		if abs(steer[i]) < 0.05:
			# near-zero steer
			if np.random.rand() < keep_prob:
				image_path_new.append(image_paths[i])
				steer_new.append(steer[i])
		else:
			image_path_new.append(image_paths[i])
			steer_new.append(steer[i])
	return image_path_new, steer_new

def flipImages(images, steer):
	images_flipped = images[:,:,::-1,:]
	steer_flipped = np.multiply(-1, steer)

	return images_flipped, steer_flipped


def generator_data(image_paths, steer, batch_size=128):
	X, y = ([], [])
	image_paths, angles = shuffle(image_paths, steer)
	while True:
		for i in range(len(steer)):
			img = cv2.imread(image_paths[i])
			angle = steer[i]
			img = preprocess(img)

			X.append(img)
			y.append(angle)

			if len(X) == batch_size:
				yield (np.array(X), np.array(y))
				X, y = ([], [])
				image_paths, angles = shuffle(image_paths, angles)
			# flip horizontally and invert steer angle, if magnitude is > 0.33
			if abs(angle) > 0.08:
				img, angle = flipImages(np.array([img]), np.array([angle]))
				X.append(img[0])
				y.append(angle[0])
				if len(X) == batch_size:
					yield (np.array(X), np.array(y))
					X, y = ([], [])
					image_paths, angles = shuffle(image_paths, angles)


def preprocess(x):
	yuv = cv2.cvtColor(x, cv2.COLOR_RGB2YUV)
	resize = cv2.resize(yuv, (160, 80))
	return yuv

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='CarND Behvioral Cloning')
	parser.add_argument('-t', action='store_true')

	args = parser.parse_args()

	image_paths, steer = readAllData(['data/', 'owndata-1/', 'owndata-reverse/', 'owndata-recovery/'])

	if args.t:
		train_gen = generator_data(image_paths, steer)
		val_gen = generator_data(image_paths, steer)

		activation_func = 'elu'

		model = Sequential()
		model.add(Lambda(lambda x: x / 130.0  - 1.0, input_shape=(160, 320, 3)))
		model.add(Cropping2D(cropping=((50, 20), (0, 0))))
		model.add(Convolution2D(24, 5, 5, activation=activation_func, border_mode='valid', subsample=(2, 2)))
		model.add(Convolution2D(36, 5, 5, activation=activation_func, border_mode='valid', subsample=(2, 2)))
		model.add(Convolution2D(48, 5, 5, activation=activation_func, border_mode='valid', subsample=(2, 2)))
		model.add(Convolution2D(64, 3, 3, border_mode='valid', activation=activation_func))
		model.add(Convolution2D(64, 3, 3, border_mode='valid', activation=activation_func))
		model.add(Flatten())
		model.add(Dropout(0.5))
		model.add(Dense(100, activation=activation_func))
		model.add(Dense(50, activation=activation_func))
		model.add(Dense(10, activation=activation_func))
		model.add(Dense(1))

		model.compile(loss='mse', optimizer='adam')

		print("FITTING")
		history = model.fit_generator(train_gen, validation_data=val_gen, nb_val_samples=8000, samples_per_epoch=40000, nb_epoch=1, verbose=1)

		print("SAVING MODEL")
		model.save('model.h5')
		print("SAVED")
