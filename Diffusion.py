
import Benchmark
import numpy as np
import keras
from keras import layers
import tensorflow as tf
from datasets import load_dataset
import math

keras.config.set_dtype_policy("mixed_bfloat16")


class DiffusionDataset(keras.utils.PyDataset):
	def __init__(self, base_images, base_labels, batch_size, np_rng=None, partial_batches=True, **kwargs):
		super().__init__(**kwargs)

		self.rng = np.random.default_rng() if np_rng is None else np_rng
		self.base_images = base_images
		self.base_labels = base_labels
		self.batch_size = batch_size
		self.partial_batches = partial_batches

	def __len__(self):
		steps_f = self.base_images.shape[0] / self.batch_size
		return math.ceil(steps_f) if self.partial_batches else math.floor(steps_f)

	def __getitem__(self, idx):

		lower = idx * self.batch_size
		upper = min(lower + self.batch_size, len(self.base_images))

		images = self.base_images[lower:upper, :, :, :]
		labels = self.base_labels[lower:upper, :,]
		shape = images.shape

		noise = self.rng.normal(0, 0.5, size=shape)
		noise = 0.5 * (np.tanh(noise) + 1)

		strength = self.rng.random((shape[0],))
		w = strength[:, None, None, None]

		lerped = noise*w + images*(1-w)

		return (lerped, labels, strength), noise


class Diffusion(Benchmark.Benchmark):
	def __init__(self, amount_encoding_dim=32, class_encoding_dim=32, conditioning_dim=64, unet_depth=2, unet_base_patterns=32, dataset="cifar"):
		super().__init__()
		self.amount_encoding_dim = amount_encoding_dim
		self.class_encoding_dim  = class_encoding_dim
		self.conditioning_dim    = conditioning_dim
		self.unet_depth          = unet_depth
		self.unet_base_patterns  = unet_base_patterns

		self.dataset = dataset


	def _data(self):
		match self.dataset:
			case "cifar":
				(x_train, l_train), (x_valid, l_valid) = keras.datasets.cifar10.load_data()

			case "resisc":
				ds_train = load_dataset("timm/resisc45", split="train").with_format("numpy")
				ds_valid = load_dataset("timm/resisc45", split="test").with_format("numpy")

				x_train =                ds_train["image"][:]
				l_train = np.expand_dims(ds_train["label"][:], -1)
				x_valid =                ds_valid["image"][:]
				l_valid = np.expand_dims(ds_valid["label"][:], -1)

			case _:
				raise ValueError(f"Dataset {self.dataset} unknown")

		x_train = x_train.astype("float32") / 255
		x_valid = x_valid.astype("float32") / 255

		self.l_train = l_train
		self.x_train = x_train
		self.l_valid = l_valid
		self.x_valid = x_valid

	def _init(self, use_xla):
		lut = {
			"cifar" : (10, ( 32,  32, 3)),
			"resisc": (45, (256, 256, 3)),
		}
		num_classes, input_shape = lut[self.dataset]

		def down_up(iterations, filters, x, conditioning):
			def condition(y):
				cond_img  = layers.Dense(filters, use_bias=False)(y)
				cond_cond = layers.Dense(filters, use_bias=True )(conditioning)
				cond_cond = layers.Reshape((1,1,filters))(cond_cond)
				y = layers.Add()([cond_cond, cond_img])
				return layers.Activation("leaky_relu")(y)

			def conv(y):
				y = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(y)
				y = layers.GroupNormalization(groups=min(8, filters))(y)
				return layers.Activation("leaky_relu")(y)

			def attn(y, heads=4):
				positions = y.shape[1] * y.shape[2]

				if positions > 2048: return y

				z = layers.LayerNormalization()(y)
				z = layers.MultiHeadAttention(heads, y.shape[-1] // heads)(z, z)
				z = layers.Add()([y, z])

				return z

			x = conv(x)
			x = condition(x)
			x = conv(x)
			x = attn(x)

			if iterations == 0: return x

			y = layers.MaxPooling2D(pool_size=(2, 2))(x)
			y = down_up(iterations-1, filters*2, y, conditioning)
			y = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(y)

			x = layers.Concatenate(axis=-1)([y, x])
			x = conv(x)
			x = condition(x)
			x = conv(x)
			x = attn(x)

			return x

		def sinusoidal(x, dim):
			half = dim // 2
			freqs = np.exp(np.linspace(0, np.log(10000), half))
			x = x * freqs[None, :]
			return layers.Concatenate()([
				layers.Lambda(tf.sin)(x),
				layers.Lambda(tf.cos)(x)
			])

		# conditioning
		input_noise_amount = layers.Input(shape=(1,))
		amount = sinusoidal(input_noise_amount, self.amount_encoding_dim)
		amount = layers.Dense(self.amount_encoding_dim, activation="leaky_relu")(amount)

		input_class = layers.Input(shape=(1,), dtype="int32")
		label = layers.Embedding(num_classes, self.class_encoding_dim)(input_class)
		label = layers.Reshape((self.class_encoding_dim,))(label)

		conditioning = layers.Concatenate(axis=-1)([label, amount])
		conditioning = layers.Dense(self.conditioning_dim, activation="leaky_relu")(conditioning)

		input_image = layers.Input(shape=input_shape)

		x = down_up(self.unet_depth, self.unet_base_patterns, input_image, conditioning)
		x = layers.Conv2D(input_shape[-1], kernel_size=(3, 3), padding="same", activation="sigmoid")(x)

		model = keras.Model(inputs=[input_image, input_class, input_noise_amount], outputs=x)

		model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"], jit_compile=use_xla)

		self.model = model

	def _info(self):
		self.model.summary()

	def _train(self, num_epochs=16):
		batch_size = {
			"cifar": 512,
			"resisc": 48,
		}[self.dataset]

		ds_train = DiffusionDataset(self.x_train, self.l_train, batch_size)
		ds_valid = DiffusionDataset(self.x_valid, self.l_valid, batch_size)

		tb_callback = keras.callbacks.TensorBoard(log_dir="./logs", update_freq=1)
		early_stop = keras.callbacks.EarlyStopping(
			monitor="val_loss",
			patience=3,
			restore_best_weights=True,
			verbose=1,
		)

		self.model.fit(
			ds_train, validation_data=ds_valid,
			epochs=num_epochs,
			callbacks=[
				keras.callbacks.TensorBoard(log_dir="./logs", update_freq=1),
				keras.callbacks.EarlyStopping(
					monitor="val_loss",
					patience=3,
					restore_best_weights=True,
					verbose=1,
				),
				keras.callbacks.ModelCheckpoint(
					filepath="./checkpoints/diffusion-{epoch:03d}.weights.h5",
					save_weights_only=True,
					save_best_only=False,
					verbose=1,
				)
			]
		)

		score = self.model.evaluate(
			ds_valid,
			verbose=0
		)
		print("Test loss:", score[0])
		print("Test accuracy:", score[1])

	def _eval(self):
		ds_valid = DiffusionDataset(self.x_valid, self.l_valid, 16, partial_batches=False)
		v = self.model.evaluate(ds_valid, verbose=0)
		print("Got", v)

	def _save(self, filename): self.model.save_weights(filename)
	def _load(self, filename): self.model.load_weights(filename)

	def _play(self):
		import matplotlib.pyplot as plt

		steps = 200

		label = np.array([[int(input("Pick a class: "))]], dtype="uint8")

		rng = np.random.default_rng()
		noise = rng.normal(0, 0.5, size=(32,32,3))
		noise = 0.5 * (np.tanh(noise) + 1)

		noise_left = 1 - (1/steps)

		image = noise[None, :, :, :]

		for i in range(1, steps):
			c_1 = image
			w_1 = noise_left
			w_2 = noise_left - (1/steps)

			n = self.model([c_1, label, np.array([[noise_left]])])

			w_n = (w_2-w_1)/(1-w_1)
			w_c = ( 1 -w_2)/(1-w_1)

			c_2 = n*w_n + c_1*w_c

			image = np.array(c_2)
			noise_left = w_2

			print(f"step {i} done")

		image = image[0]

		plt.imshow(image)
		plt.savefig("out.png", dpi=150, bbox_inches="tight")
