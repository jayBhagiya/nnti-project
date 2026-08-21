import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.xglm.modeling_xglm import XGLMAttention

class BitFitAdaptedModel(nn.Module):
	def __init__(self, MODEL_NAME):
		super(BitFitAdaptedModel, self).__init__()
		self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
		self.model_config = self.model.config

		self.trainable_params_patterns = ["bias"]

		# freeze the weights and allow only the biases to get updated.
		# BitFit updates only the bias terms and not the weights. 
		# The bias terms are the additive constants in various layers of 
		# the model, hence, fine tuning these parameters alone is very efficient.
		self.freeze_parameters_except_lora(self.model)

	def freeze_parameters_except_lora(self, model):
		for name, param in model.named_parameters():
			if any(pattern in name for pattern in self.trainable_params_patterns):
				param.requires_grad = True
			else:
				param.requires_grad = False

	def forward(self, *args, **kwargs):
		return self.model(*args, **kwargs)


class LoRALinear(nn.Module):
	def __init__(self, linear_layer, rank=4, alpha=1):
		super().__init__()

		self.in_features = linear_layer.in_features
		self.out_features = linear_layer.out_features
		self.rank = rank
		self.alpha = alpha
		self.weight = linear_layer.weight
		self.bias = linear_layer.bias

		# Section 4.1 of the paper:
		#   We are using a random Gaussian distribution for the initialization for A and 0 for B, so delta(W) = BA will be 0 at the starting
		self.lora_mat_A = nn.Parameter(torch.randn((self.rank, self.out_features)), requires_grad=True)
		self.lora_mat_B = nn.Parameter(torch.zeros((self.in_features, self.rank)), requires_grad=True)

		# Section 4.1 of the paper:
		#   We then scale ∆Wx by α/r , where α is a constant in R.
		#   When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately.
		#   As a result, we simply set α to the first r we try and do not tune it.
		#   This scaling helps to reduce the need to retune hyperparameters when we vary r.
		self.scale = alpha / rank
		self.enabled = True

	def forward(self, input):
		weight = self.weight
		if self.enabled:
			# W + (B*A)*scale
			weight = weight + torch.matmul(self.lora_mat_B, self.lora_mat_A).view(self.weight.shape) * self.scale
		else:
			weight = weight

		return F.linear(input, weight, self.bias)


class LoRaAdaptedModel(nn.Module):
	def __init__(self, MODEL_NAME, rank=4, alpha=1):
		super(LoRaAdaptedModel, self).__init__()
		self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
		self.model_config = self.model.config

		self.rank = rank
		self.alpha = alpha

		self.m_patterns = "_proj"
		self.trainable_params_patterns = ["lora_"]

		self.replace_linear_layers(self.model)
		self.freeze_parameters_except_lora(self.model)

	def replace_linear_layers(self, model):
		for name, module in model.named_children():
			if (self.m_patterns in name):
				if isinstance(module, nn.Linear):
					new_layer = LoRALinear(module, rank=self.rank, alpha=self.alpha)
					state_dict_old = module.state_dict()

					# Load the state dict to the new layer
					new_layer.load_state_dict(state_dict_old, strict=False)

					# Get the state of the new layer
					state_dict_new = new_layer.state_dict()

					keys_old = set(state_dict_old.keys())
					keys_new = set(k for k in state_dict_new.keys() if not k.startswith("lora_"))
					assert keys_old == keys_new, f"Keys of the state dictionaries don't match (ignoring lora_ parameters):\n\tExpected Parameters: {keys_old}\n\tNew Parameters (w.o. lora): {keys_new}"

					# Replace the original layer with the new layer
					setattr(model, name, new_layer)
			else:
				# Recurse on the child modules
				self.replace_linear_layers(module)

	def freeze_parameters_except_lora(self, model):
		for name, param in model.named_parameters():
			if any(pattern in name for pattern in self.trainable_params_patterns):
				param.requires_grad = True
			else:
				param.requires_grad = False

	def forward(self, *args, **kwargs):
		return self.model(*args, **kwargs)


class IA3AttentionLayer(XGLMAttention):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		self.ia3_l_k = nn.Parameter(torch.randn(self.k_proj.out_features), requires_grad=True)
		self.ia3_l_v = nn.Parameter(torch.randn(self.v_proj.out_features), requires_grad=True)

	def ia3_k_proj(self, x):
		"""
		Applies IA3 adaptation to the key component. Computes a modified key output by elementwise multiplication with learned vector
		to the standard value output. Requires the regular linear layer 
		to be frozen before training.
		"""
		ia3_key_states = self.ia3_l_k * self.k_proj(x)
		return ia3_key_states

	def ia3_v_proj(self, x):
		"""
		Applies IA3 adaptation to the value component. Computes a modified value output by elementwise multiplication with learned vector
		to the standard value output. Requires the regular linear layer to be frozen before training.
		"""
		ia3_value_states = self.ia3_l_v * self.v_proj(x)
		return ia3_value_states

	def forward(
		self,
		hidden_states: torch.Tensor,
		key_value_states: Optional[torch.Tensor] = None,
		past_key_value: Optional[Tuple[torch.Tensor]] = None,
		attention_mask: Optional[torch.Tensor] = None,
		layer_head_mask: Optional[torch.Tensor] = None,
		output_attentions: bool = False,
	) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:   # Function from transformer/models/xglm/modeling_xglm.py
		"""Input shape: Batch x Time x Channel"""

		# if key_value_states are provided this layer is used as a cross-attention layer
		# for the decoder
		is_cross_attention = key_value_states is not None

		bsz, tgt_len, _ = hidden_states.size()

		# get query proj
		query_states = self.q_proj(hidden_states) * self.scaling
		# get key, value proj
		if is_cross_attention and past_key_value is not None:
			# reuse k,v, cross_attentions
			key_states = past_key_value[0]
			value_states = past_key_value[1]
		elif is_cross_attention:
			# cross_attentions
			key_states = self._shape(self.ia3_k_proj(key_value_states), -1, bsz)            # Introduce element wise multiplication to keys 
			value_states = self._shape(self.ia3_v_proj(key_value_states), -1, bsz)          # Introduce element wise multiplication to values
		elif past_key_value is not None:
			# reuse k, v, self_attention
			key_states = self._shape(self.ia3_k_proj(hidden_states), -1, bsz)               # Introduce element wise multiplication to keys 
			value_states = self._shape(self.ia3_v_proj(hidden_states), -1, bsz)             # Introduce element wise multiplication to values
			key_states = torch.cat([past_key_value[0], key_states], dim=2)
			value_states = torch.cat([past_key_value[1], value_states], dim=2)
		else:
			# self_attention
			key_states = self._shape(self.ia3_k_proj(hidden_states), -1, bsz)               # Introduce element wise multiplication to keys
			value_states = self._shape(self.ia3_v_proj(hidden_states), -1, bsz)             # Introduce element wise multiplication to values

		if self.is_decoder:
			# if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
			# Further calls to cross_attention layer can then reuse all cross-attention
			# key/value_states (first "if" case)
			# if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
			# all previous decoder key/value_states. Further calls to uni-directional self-attention
			# can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
			# if encoder bi-directional self-attention `past_key_value` is always `None`
			past_key_value = (key_states, value_states)

		proj_shape = (bsz * self.num_heads, -1, self.head_dim)
		query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
		key_states = key_states.view(*proj_shape)
		value_states = value_states.view(*proj_shape)

		src_len = key_states.size(1)
		attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

		if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
			raise ValueError(
				f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
				f" {attn_weights.size()}"
			)

		if attention_mask is not None:
			if attention_mask.size() != (bsz, 1, tgt_len, src_len):
				raise ValueError(
					f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
				)
			attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
			attn_weights = torch.max(
				attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
			)
			attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

		# upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
		if attn_weights.dtype == torch.float16:
			attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
		else:
			attn_weights = nn.functional.softmax(attn_weights, dim=-1)

		if layer_head_mask is not None:
			if layer_head_mask.size() != (self.num_heads,):
				raise ValueError(
					f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
					f" {layer_head_mask.size()}"
				)
			attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
			attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

		if output_attentions:
			# this operation is a bit awkward, but it's required to
			# make sure that attn_weights keeps its gradient.
			# In order to do so, attn_weights have to be reshaped
			# twice and have to be reused in the following
			attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
			attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
		else:
			attn_weights_reshaped = None

		attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

		attn_output = torch.bmm(attn_probs, value_states)

		if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
			raise ValueError(
				f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
				f" {attn_output.size()}"
			)

		attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
		attn_output = attn_output.transpose(1, 2)

		# Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
		# partitioned aross GPUs when using tensor-parallelism.
		attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

		attn_output = self.out_proj(attn_output)

		return attn_output, attn_weights_reshaped, past_key_value
	
class IA3DenseLayer(nn.Linear):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		self.ia3_l_ff = nn.Parameter(torch.randn(self.out_features), requires_grad=True)

	def forward(self, x):
		Wx = F.linear(x, self.weight, self.bias)
		return self.ia3_l_ff * Wx
	
class IA3AdaptedModel(nn.Module):
	"""Class for the Classification model. It has Efficientnet B0 model as the basemodel with fully connected layer head"""

	def __init__(self, MODEL_NAME):
		super(IA3AdaptedModel, self).__init__()
		self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
		self.model_config = self.model.config

		self.replace_attention_layers(self.model)
		self.add_ia3_dense_layers(self.model)

		# Freeze whole model except ia3_parameters

		for name, param in self.model.named_parameters():
			if ("ia3_" in name):
				param.requires_grad = True
			else:
				param.requires_grad = False

	def replace_attention_layers(self, model):
		for name, module in model.named_children():
			if isinstance(module, XGLMAttention):

				# Create a new LoraMultiheadAttention layer
				new_layer = IA3AttentionLayer(
											embed_dim = module.embed_dim,
											num_heads= module.num_heads,
											dropout= module.dropout,
											is_decoder= module.is_decoder,
										)

				# Get the state of the original layer
				state_dict_old = module.state_dict()

				# Load the state dict to the new layer
				new_layer.load_state_dict(state_dict_old, strict=False)

				# Get the state of the new layer
				state_dict_new = new_layer.state_dict()

				# Compare keys of both state dicts
				keys_old = set(state_dict_old.keys())
				keys_new = set(k for k in state_dict_new.keys() if not k.startswith("ia3_"))
				assert keys_old == keys_new, f"Keys of the state dictionaries don't match (ignoring ia3_ parameters):\n\tExpected Parameters: {keys_old}\n\tNew Parameters (w.o. IA3): {keys_new}"

				# Replace the original layer with the new layer
				setattr(model, name, new_layer)

			else:
				# Recurse on the child modules
				self.replace_attention_layers(module)
		
	def add_ia3_dense_layers(self, model):
		for name, module in model.named_children():
			if name == 'fc1':
				new_layer = IA3DenseLayer(module.in_features, module.out_features, bias=True)
				state_dict_old = module.state_dict()

				# Load the state dict to the new layer
				new_layer.load_state_dict(state_dict_old, strict=False)

				# Get the state of the new layer
				state_dict_new = new_layer.state_dict()

				keys_old = set(state_dict_old.keys())
				keys_new = set(k for k in state_dict_new.keys() if not k.startswith("ia3_"))
				assert keys_old == keys_new, f"Keys of the state dictionaries don't match (ignoring ia3_ parameters):\n\tExpected Parameters: {keys_old}\n\tNew Parameters (w.o. IA3): {keys_new}"

				# Replace the original layer with the new layer
				setattr(model, name, new_layer)

			else:
				# Recurse on the child modules
				self.add_ia3_dense_layers(module)

	def forward(self, *args, **kwargs):
		return self.model(*args, **kwargs)