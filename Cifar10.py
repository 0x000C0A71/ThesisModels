
import Benchmark
import numpy as np
import keras
from keras import layers


class Cifar10(Benchmark.Benchmark):
	def _data(self):
		num_classes = 10

		(x_trai, y_trai), (x_test, y_test) = keras.datasets.cifar10.load_data()

		x_trai = x_trai.astype("float32") / 255
		x_test = x_test.astype("float32") / 255
		y_trai = keras.utils.to_categorical(y_trai, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes)

		self.x_trai = x_trai
		self.y_trai = y_trai
		self.x_test = x_test
		self.y_test = y_test


	def _init(self, use_xla):
		num_classes = 10
		input_shape = (32, 32, 3)

		model = keras.Sequential(
			# [ keras.Input(shape=input_shape)
			# , layers.Conv2D(16, kernel_size=(3, 3), activation="leaky_relu", padding="same")
			# , layers.Conv2D(32, kernel_size=(3, 3), activation="leaky_relu", padding="same")
			# , layers.AveragePooling2D(pool_size=(2, 2))
			# , layers.Conv2D(48, kernel_size=(3, 3), activation="leaky_relu", padding="same")
			# , layers.AveragePooling2D(pool_size=(2, 2))
			# , layers.Conv2D(64, kernel_size=(3, 3), activation="leaky_relu", padding="same")
			# , layers.AveragePooling2D(pool_size=(2, 2))
			# , layers.Flatten()
			# #, layers.Dropout(0.5)
			# , layers.Dense(num_classes, activation="softmax")
			# ]
			#[ keras.Input(shape=input_shape)
			#, layers.Conv2D(32, kernel_size=(3, 3), activation="leaky_relu")
			#, layers.MaxPooling2D(pool_size=(2, 2))
			#, layers.Conv2D(64, kernel_size=(3, 3), activation="leaky_relu")
			#, layers.MaxPooling2D(pool_size=(2, 2))
			#, layers.Flatten()
			##, layers.Dropout(0.5)
			#, layers.Dense(num_classes, activation="softmax")
			#]
			[ keras.Input(shape=input_shape)
			, layers.Conv2D( 4, kernel_size=(3, 3), activation="leaky_relu")
			, layers.Conv2D( 8, kernel_size=(3, 3), activation="leaky_relu")
			, layers.Conv2D(12, kernel_size=(3, 3), activation="leaky_relu")
			, layers.Conv2D(16, kernel_size=(3, 3), activation="leaky_relu")
			, layers.Dropout(0.3)
			, layers.Conv2D(20, kernel_size=(3, 3), activation="leaky_relu")
			, layers.Conv2D(24, kernel_size=(3, 3), activation="leaky_relu")
			, layers.Conv2D(28, kernel_size=(3, 3), activation="leaky_relu")
			, layers.Conv2D(32, kernel_size=(3, 3), activation="leaky_relu")
			, layers.Dropout(0.3)
			, layers.Conv2D(36, kernel_size=(3, 3), activation="leaky_relu")
			, layers.Conv2D(40, kernel_size=(3, 3), activation="leaky_relu")
			, layers.Conv2D(44, kernel_size=(3, 3), activation="leaky_relu")
			, layers.Conv2D(48, kernel_size=(3, 3), activation="leaky_relu")
			, layers.Dropout(0.3)
			, layers.Conv2D(52, kernel_size=(3, 3), activation="leaky_relu")
			, layers.Conv2D(56, kernel_size=(3, 3), activation="leaky_relu")
			, layers.Conv2D(60, kernel_size=(3, 3), activation="leaky_relu")
			, layers.Conv2D(64, kernel_size=(2, 2), activation="leaky_relu")
			, layers.Flatten()
			, layers.Dropout(0.5)
			, layers.Dense(num_classes, activation="softmax")
			]
		)

		model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"], jit_compile=use_xla)

		self.model = model

	def _info(self):
		self.model.summary()

	def _train(self, num_epochs=16):

		tb_callback = keras.callbacks.TensorBoard(log_dir="./logs", update_freq=1)

		self.model.fit(self.x_trai, self.y_trai, batch_size=128, epochs=num_epochs, validation_split=0.1)

		score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
		print("Test loss:", score[0])
		print("Test accuracy:", score[1])

	def _save(self, filename): self.model.save_weights(filename)
	def _load(self, filename): self.model.load_weights(filename)

	def _eval(self):
		v = self.model.evaluate(self.x_test, self.y_test, verbose=0, batch_size=16)
		print("Got", v)

