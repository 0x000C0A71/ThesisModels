
import Benchmark
import numpy as np
import keras
from keras import layers
from datasets import load_dataset
import math


import tensorflow as tf

keras.config.set_dtype_policy("mixed_bfloat16")

def load_wikipedia(n):
	dataset = load_dataset(
		"wikimedia/wikipedia",
		"20231101.en",
		split="train",
		streaming=True
	)
	articles = []
	for i, example in enumerate(dataset):
		if i >= n: break
		articles.append(example["text"])
	return articles

class AddPositionalEmbedding(layers.Layer):
	def __init__(self, context_length, dim, **kwargs):
		super().__init__(**kwargs)
		self.context_length = context_length
		self.pos_emb = layers.Embedding(context_length, dim)

	def call(self, x):
		seq_len = tf.shape(x)[1]
		positions = tf.range(seq_len)
		return x + self.pos_emb(positions)

class TokenDataset(keras.utils.PyDataset):
	def __init__(self, tokens, context_length, batch_size, model_param_count=None, steps_per_epoch=None, **kwargs):
		super().__init__(**kwargs)
		self.tokens          = tokens
		self.context_length  = context_length
		self.batch_size      = batch_size

		match (model_param_count, steps_per_epoch):
			case (None, None):
				raise ValueError("Either model_param_count or steps_per_epoch must be passed")

			case (params, None):
				training_tokens = params*20
				float_steps = training_tokens/context_length/batch_size
				self.steps_per_epoch = math.ceil(float_steps)
				if training_tokens > len(tokens): print(f"Warning: for a model with {params} parameters, at least {training_tokens} tokens should be trained on, but only {len(tokens)} were supplied")

			case (None, steps):
				self.steps_per_epoch = steps

			case (_, _):
				raise ValueError("Only one of model_param_count or steps_per_epoch can be passed")


	def __len__(self):
		return self.steps_per_epoch

	def __getitem__(self, idx):
		n = len(self.tokens)
		starts = np.random.randint(0, n - self.context_length - 1, size=self.batch_size)
		x = np.stack([self.tokens[s     : s + self.context_length    ] for s in starts])
		y = np.stack([self.tokens[s + 1 : s + self.context_length + 1] for s in starts])
		return x.astype("int32"), y.astype("int32")

class GPT(Benchmark.Benchmark):
	def __init__(self, n_articles=256, context_length=1024, token_latent_dim=96, attention_heads=6, transformer_layers=8):
		super().__init__()
		self.n_articles         = n_articles
		self.context_length     = context_length
		self.token_latent_dim   = token_latent_dim
		self.attention_heads    = attention_heads
		self.transformer_layers = transformer_layers


	def _data(self):
		corpus = load_wikipedia(self.n_articles + 2)

		def tokenize(corpus, delimiter=0):
			buff = bytearray()
			for text in corpus:
				buff.append(delimiter)
				buff.extend(text.encode())
			return np.frombuffer(buff, dtype="uint8")

		self.train_tokens = tokenize(corpus[:-2])
		self.valid_tokens = tokenize(corpus[-2:])

	def _init(self, use_xla):
		num_tokens = 256

		def transformer(x):
			y = layers.LayerNormalization()(x)
			y = layers.MultiHeadAttention(self.attention_heads, self.token_latent_dim // self.attention_heads)(
				query=y, value=y,
				use_causal_mask=True,
			)
			x = layers.Add()([x, y])

			ffn = layers.LayerNormalization()(x)
			ffn = layers.Dense(4*self.token_latent_dim, activation="gelu")(ffn)
			ffn = layers.Dense(  self.token_latent_dim)(ffn)

			x = layers.Add()([x, ffn])
			return x

		inputs = layers.Input(shape=(self.context_length,), dtype="int32")

		x = layers.Embedding(num_tokens, self.token_latent_dim)(inputs)
		x = AddPositionalEmbedding(self.context_length, self.token_latent_dim)(x)
		for _ in range(self.transformer_layers): x = transformer(x)
		x = layers.LayerNormalization()(x)
		x = layers.Dense(num_tokens, activation="softmax")(x)

		model = keras.Model(inputs=inputs, outputs=x)

		model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"], jit_compile=use_xla)

		self.model = model

	def _info(self):
		self.model.summary()

	def _train(self, num_epochs=20):
		#batch_size = 40
		batch_size = 28

		ds_train = TokenDataset(self.train_tokens, self.context_length, batch_size, model_param_count=self.model.count_params())
		ds_valid = TokenDataset(self.valid_tokens, self.context_length, batch_size, steps_per_epoch=10)

		tb_callback = keras.callbacks.TensorBoard(log_dir="./logs", update_freq=1)
		early_stop = keras.callbacks.EarlyStopping(
			monitor="val_loss",
			patience=3,
			restore_best_weights=True,
			verbose=1,
		)

		self.model.fit(
			ds_train, validation_data=ds_valid,
			epochs=num_epochs, callbacks=[tb_callback, early_stop]
		)

		score = self.model.evaluate(ds_valid, verbose=0)
		print("Test loss:", score[0])
		print("Test accuracy:", score[1])

	def _eval(self):
		batch_size = 16

		ds_valid = TokenDataset(self.valid_tokens, self.context_length, batch_size, steps_per_epoch=100)
		v = self.model.evaluate(ds_valid, verbose=0)
		print("Got", v)

	def _save(self, filename): self.model.save_weights(filename)
	def _load(self, filename): self.model.load_weights(filename)

	def _play(self):
		tokens_to_generate = 20
		sample_random = False

		starting_point = input("Type a text for the language model to complete: ")

		tokens = np.array([x for x in starting_point.encode()], dtype="uint8")

		current_index = len(tokens)+1
		tokens = np.pad(tokens, (1, 1024-current_index), "constant", constant_values=0)


		for _ in range(tokens_to_generate):
			predicted = np.array(self.model(tokens[None,:])[0,current_index])

			predicted = predicted / predicted.sum()
			#print("distribution:", predicted)

			current_index += 1

			top_index = np.argmax(predicted)
			top_prob = predicted[top_index]

			print(f"top pick: {top_index} ('{chr(top_index)}') with prob {top_prob}")

			next_token = top_index

			if sample_random:
				sampled_index = np.random.choice(256, p=predicted)
				sampled_prob = predicted[sampled_index]

				print(f"sampled: {sampled_index} ('{chr(sampled_index)}') with prob {sampled_prob}")

				next_token = sampled_index

			tokens[current_index] = next_token

		final_output = bytes(tokens).decode()
		print("Final text:", final_output)






