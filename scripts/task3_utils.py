import math
import numpy as np

class ParamsUtils():
	def __init__(self, model, peft_model, name):
		super(ParamsUtils, self).__init__()
		self.model = model
		self.peft_model = peft_model
		self.name = name

		self.total_pre_peft_params = self.count_params(self.model, True)

		self.total_peft_params = self.count_params(self.peft_model, True)
		self.total_non_peft_params = self.count_params(self.peft_model, False)
		self.total_params = self.total_peft_params + self.total_non_peft_params

	def count_params(self, model, if_requires_grad):
		if if_requires_grad:
			return sum(p.numel() for p in model.parameters() if p.requires_grad)
		else:
			return sum(p.numel() for p in model.parameters() if not p.requires_grad)

	def print_stats(self):
		# The non-peft parameters count must match the original network
		print("=================================================================")
		print(f"Parameter counts for the {self.name}")
		# assert self.total_non_peft_params == self.total_pre_peft_params, f"The non-peft parameters count must match the original network parameters:\n\tExpected non-peft Parameters: {self.total_non_peft_params}\n\toriginal Parameters: {self.total_pre_peft_params}"
		print(f'Total number of parameters (original): {self.total_non_peft_params:,}')
		print(f'Total number of parameters (original + {self.name}): {self.total_params:,}')
		print(f'Parameters introduced by {self.name}: {self.total_peft_params:,}')
		parameters_incremment = (self.total_peft_params / self.total_non_peft_params) * 100
		print(f'Parameters incremment: {parameters_incremment:.3f}%')
		print("=================================================================")
