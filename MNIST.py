
import Benchmark
import numpy as np
import keras
from keras import layers


class MNIST(Benchmark.Benchmark):
	def _data(self):
		num_classes = 10

		(x_train, y_train), (x_valid, y_valid) = keras.datasets.mnist.load_data()

		x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
		x_valid = np.expand_dims(x_valid.astype("float32") / 255, -1)

		self.x_train = x_train
		self.y_train = y_train
		self.x_valid = x_valid
		self.y_valid = y_valid


	def _init(self, use_xla):
		num_classes = 10
		input_shape = (28, 28, 1)

		model = keras.Sequential(
			[ keras.Input(shape=input_shape)
			, layers.Conv2D(16, kernel_size=(3, 3), activation="leaky_relu")
			, layers.MaxPooling2D(pool_size=(2, 2))
			, layers.Conv2D(32, kernel_size=(3, 3), activation="leaky_relu")
			, layers.MaxPooling2D(pool_size=(2, 2))
			, layers.Flatten()
			#, layers.Dropout(0.5)
			, layers.Dense(num_classes, activation="softmax")
			]
		)

		model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"], jit_compile=use_xla)

		self.model = model

	def _info(self):
		self.model.summary()

	def _train(self, num_epochs=16):

		self.model.fit(
			self.x_train, self.y_train,
			validation_data=(self.x_valid, self.y_valid),
			batch_size=128, epochs=num_epochs,
			callbacks=[
				keras.callbacks.TensorBoard(log_dir="./logs", update_freq=1),
				keras.callbacks.EarlyStopping(
					monitor="val_loss",
					patience=3,
					restore_best_weights=True,
					verbose=1,
				),
			]
		)

		score = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)
		print("Test loss:", score[0])
		print("Test accuracy:", score[1])

	def _save(self, filename): self.model.save_weights(filename)
	def _load(self, filename): self.model.load_weights(filename)

	def _eval(self):
		v = self.model.evaluate(self.x_valid, self.y_valid, verbose=0, batch_size=16)
		print("Got", v)

