
import os

class Benchmark:
	def _init (self, use_xla) : raise Exception("Function must be overloaded: _init (self, use_xla) ")
	def _train(self)          : raise Exception("Function must be overloaded: _train(self)          ")
	def _eval (self)          : raise Exception("Function must be overloaded: _eval (self)          ")
	def _data (self)          : raise Exception("Function must be overloaded: _data (self)          ")
	def _info (self)          : raise Exception("Function must be overloaded: _info (self)          ")
	def _save (self, filename): raise Exception("Function must be overloaded: _save (self, filename)")
	def _load (self, filename): raise Exception("Function must be overloaded: _load (self, filename)")
	def _play (self)          : raise Exception("This model does not implement play functionality   ")

	def __init__(self):
		self.inited = False
		self.data_loaded = False

	def init(self, use_xla=False):
		self._init(use_xla)
		self.inited = True

	def data(self):
		self._data()
		self.data_loaded = True

	def train(self, num_epochs=None):
		if not self.inited:      self.init()
		if not self.data_loaded: self.data()

		if num_epochs is None:
			self._train()
		else:
			self._train(num_epochs)


	def eval(self):
		if not self.inited:      self.init()
		if not self.data_loaded: self.data()

		self._eval()

	def save (self, filename):
		if not self.inited: self.init()

		self._save(filename)

		if not os.path.isfile(filename):
			raise Exception("Train did not produce weights.")

	def load (self, filename):
		if not os.path.isfile(filename): raise Exception("Weights do not exist.")
		if not self.inited: self.init()

		self._load(filename)

	def info(self):
		if not self.inited: self.init()
		self._info()

	def play(self):
		if not self.inited: self.init()

		self._play(filename)


