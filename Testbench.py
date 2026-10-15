
import sys


import tensorflow as tf

for gpu in tf.config.list_physical_devices('GPU'):
	tf.config.experimental.set_memory_growth(gpu, True)
	tf.config.set_logical_device_configuration(gpu, [
		tf.config.LogicalDeviceConfiguration(memory_limit=13588) # 12 GB
	])

from Cifar10 import Cifar10
from UNet import UNet
from GPT import GPT
from Diffusion import Diffusion
from MNIST import MNIST


benchmarks = {
	"mnist": MNIST(),
	"cifar10": Cifar10(),
	"unet": UNet(),
	"diffusion": Diffusion(),
	"diffusion-large": Diffusion(
		dataset="resisc",
		unet_depth=4,
		amount_encoding_dim=64,
		class_encoding_dim=64,
		conditioning_dim=128
	),
	"gpt-small": GPT(),
	"gpt": GPT(token_latent_dim=128, attention_heads=8),
	"gpt-large": GPT(token_latent_dim=256, attention_heads=16),
}

weights_for = lambda name: f"./weights/{name}.weights.h5"

if __name__ == "__main__":
	match sys.argv:
		case [_, "--help"]:
			print("- <command>                     : Train all models")
			print("- <command> --quick             : Train all models, but only 1 epoch each")
			print("- <command> --random            : Initialize all models, saving their weights without training")
			print("- <command> <model name>        : evaluate specific model")
			print("- <command> --help              : show this")
			print("- <command> --models            : list all available models")
			print("- <command> --full <model name> : do info -> train -> eval for a specific model")
			print("- <command> --info <model name> : init specified model and give info")
			print("- <command> --play <model name> : interact with the specified model. Not supported by all models")

		case [_]:
			print("Training all models.")
			for (name, bench) in benchmarks.items():
				print(f"Training {name}")
				bench.train()
				bench.save(weights_for(name))

		case [_, "--quick"]:
			print("Training all models. (quickly)")
			for (name, bench) in benchmarks.items():
				bench.train(num_epochs=1)
				bench.save(weights_for(name))

		case [_, "--random"]:
			print("Initializing and savin all models")
			for (name, bench) in benchmarks.items():
				bench.save(weights_for(name))

		case [_, "--models"]:
			for name in benchmarks.keys(): print(name)

		case [_, "--full", name]:
			print(f"Doing both for only {name}")
			benchmarks[name].info()
			benchmarks[name].train()
			benchmarks[name].save(weights_for(name))
			benchmarks[name].eval()

		case [_, "--info", name]:
			print(f"Infoing only {name}")
			benchmarks[name].info()

		case [_, "--play", name]:
			print(f"Interacting with {name}")
			benchmarks[name].load(weights_for(name))
			benchmarks[name].play()

		case [_, name]:
			print(f"Only evaluating {name}")
			benchmarks[name].init(use_xla=True)
			benchmarks[name].load(weights_for(name))
			benchmarks[name].eval()

		case []: raise Exception("How did we get here?")
		case _ : raise Exception("Passed too many arguments")




