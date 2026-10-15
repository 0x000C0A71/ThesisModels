
import Benchmark
import numpy as np
import keras
from keras import layers


class UNet(Benchmark.Benchmark):

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

		def down_up(iterations, filters, x):
			x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same", activation="leaky_relu")(x)
			x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same", activation="leaky_relu")(x)

			if iterations == 0: return x

			y = layers.MaxPooling2D(pool_size=(2, 2))(x)
			y = down_up(iterations-1, filters*2, y)
			y = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(y)

			x = layers.Concatenate(axis=-1)([y, x])
			x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same", activation="leaky_relu")(x)
			x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same", activation="leaky_relu")(x)

			return x

		inputs = layers.Input(shape=input_shape)
		x = down_up(2, 64, inputs)
		x = layers.Conv2D(num_classes, kernel_size=(3, 3), padding="same", activation="leaky_relu")(x)

		x = layers.GlobalAveragePooling2D()(x)
		x = layers.Softmax()(x)

		model = keras.Model(inputs=inputs, outputs=x)

		model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"], jit_compile=use_xla)

		self.model = model

	def _info(self):
		self.model.summary()

	def _train(self, num_epochs=16):

		tb_callback = keras.callbacks.TensorBoard(log_dir="./logs", update_freq=1)

		self.model.fit(
			self.x_trai, self.y_trai,
			batch_size=512, epochs=num_epochs,
			validation_split=0.1,
			callbacks=[tb_callback]
		)

		score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
		print("Test loss:", score[0])
		print("Test accuracy:", score[1])

	def _eval(self):
		v = self.model.evaluate(self.x_test, self.y_test, verbose=0, batch_size=16)
		print("Got", v)

	def _save(self, filename): self.model.save_weights(filename)
	def _load(self, filename): self.model.load_weights(filename)
