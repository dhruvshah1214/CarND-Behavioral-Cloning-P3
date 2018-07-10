import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import argparse

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Lambda, Dropout, Cropping2D
from keras.regularizers import l2
import tensorflow as tf
from keras.preprocessing import image as keras_image
from keras.optimizers import Adam

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
			if 'recovery' in root_path and abs(float(line[3])) < 0.0:
				continue	
			
			if root_path == 'data/' and abs(float(line[3])) > 1.0:
				continue

			path = line[0]
			filename = path.split('/')[-1]
			# print(filename)
			full_path = root_path + 'IMG/' + filename
			# print(full_path)
			measurement_steer = float(line[3])

			image_paths.append(full_path)
			steer.append(measurement_steer)

			steer_correction = 0.2

			if measurement_steer > 0.3:
				path_left = root_path + 'IMG/' + line[1].split('/')[-1]

				image_paths.append(path_left)
				steer.append(measurement_steer + steer_correction)

			if measurement_steer < -0.3:
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
		if abs(float(steer[i])) < 0.1 or steer[i] < -1.95:
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


def generator_data(image_paths, steer, batch_size=64, keep_probs=[]):
	X, y = ([], [])
	image_paths, angles = shuffle(image_paths, steer)
	while True:
		for i in range(len(steer)):
			img = cv2.cvtColor(cv2.imread(image_paths[i]), cv2.COLOR_BGR2RGB)
			angle = steer[i]
			img = preprocess(img)

			X.append(img)
			y.append(angle)

			if len(X) == batch_size:
				yield (np.array(X), np.array(y))
				X, y = ([], [])
				image_paths, angles = shuffle(image_paths, angles)


			for j in range(num_bins):
				if angle > bins[j] and angle <= bins[j+1]:
					if np.random.rand() < keep_probs[j] - 1.0:
						X.append(random_brightness(img))
						y.append(angle)
						break
			if len(X) == batch_size:
				yield (np.array(X), np.array(y))
				X, y = ([], [])
				image_paths, angles = shuffle(image_paths, angles)

			# flip horizontally and invert steer angle, if magnitude is > 0.33
			if abs(float(angle)) > 0.3:
				img = img[:,::-1,:]
				angle = -1.0 * angle
				X.append(img)
				y.append(angle)
				if len(X) == batch_size:
					yield (np.array(X), np.array(y))
					X, y = ([], [])
					image_paths, angles = shuffle(image_paths, angles)

def random_brightness(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	rand = random.uniform(0.3,1.0)
	hsv[:,:,2] = rand*hsv[:,:,2]
	new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	return new_img 


def preprocess(x):
	new_img = x[60:140,:,:]
	new_img = cv2.GaussianBlur(new_img, (3,3), 0)
	new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
	new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2YUV)
	return new_img

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='CarND Behvioral Cloning')
	parser.add_argument('-t', action='store_true')

	args = parser.parse_args()

	image_paths, angles = readAllData(['owndata-reverse/', 'owndata-3/', 'owndata-recovery6/', 'owndata-recovery5/', 'owndata-recovery4/', 'owndata-recovery8/'])

	#image_paths, steer = lowerZeroes(image_paths, steer, keep_prob=0.1)

	num_bins = 23
	avg_samples_per_bin = len(angles)/num_bins
	hist, bins = np.histogram(angles, num_bins)
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	keep_probs = []
	target = avg_samples_per_bin * 0.5
	for i in range(num_bins):
		keep_probs.append(1./(hist[i]/target))
	remove_list = []
	for i in range(len(angles)):
		for j in range(num_bins):
			if angles[i] > bins[j] and angles[i] <= bins[j+1]:	
				if np.random.rand() > keep_probs[j]:
					remove_list.append(i)
	image_paths = np.delete(image_paths, remove_list, axis=0)
	angles = np.delete(angles, remove_list)

	hist, bins = np.histogram(angles, num_bins)
	plt.bar(center, hist, align='center', width=width)
	plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
	plt.show()
	

	if args.t:
		train_gen = generator_data(image_paths, steer, batch_size=128, keep_probs=(keep_probs, bins))

		activation_func = 'elu'

		model = Sequential()
		model.add(Lambda(lambda x: x / 127.5  - 1.0, input_shape=(66, 200, 3)))
		model.add(Convolution2D(24, 5, 5, activation=activation_func, border_mode='valid', subsample=(2, 2)))
		model.add(Convolution2D(36, 5, 5, activation=activation_func, border_mode='valid', subsample=(2, 2)))
		model.add(Convolution2D(48, 5, 5, activation=activation_func, border_mode='valid', subsample=(2, 2)))
		model.add(Convolution2D(64, 3, 3, border_mode='valid', activation=activation_func))
		model.add(Convolution2D(64, 3, 3, border_mode='valid', activation=activation_func))
		model.add(Flatten())
		model.add(Dropout(0.5))
		model.add(Dense(100))
		model.add(Dense(50))
		model.add(Dense(10))
		model.add(Dense(1))
		
		adam = Adam(lr=0.0001)
		model.compile(loss='mse', optimizer=adam)
		print(model.summary())
		print("FITTING")
		history = model.fit_generator(train_gen, samples_per_epoch=20000, nb_epoch=1, verbose=1)

		print("SAVING MODEL")
		model.save('model.h5')
		print("SAVED")
