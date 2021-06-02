import torch

class FallacyDataset:
	def __init__(self, texts, binary, multi, label_encoder_binary, label_encoder_multi,weights_multi_class, weights_binary_class,weights_multi_seq, weights_binary_seq, params):

		self.texts = texts
		self.binary = binary
		self.multi = multi
		self.label_encoder_binary = label_encoder_binary
		self.label_encoder_multi = label_encoder_multi
		self.weights_multi_class = weights_multi_class
		self.weights_binary_class = weights_binary_class
		self.weights_multi_seq = weights_multi_seq
		self.weights_binary_seq = weights_binary_seq
		self.params_a = params[0]
		self.params_b = params[1]

	def __len__(self):
		return len(self.texts)


	def __getitem__(self, item):
		text = self.texts[item]
		binary = self.binary[item]
		multi = self.multi[item]
		label_encoder_binary = self.label_encoder_binary
		label_encoder_multi = self.label_encoder_multi

		binary_class = list(set(binary))
		multi_class = list(set(multi))

		assert len(binary_class) <= 2 and len(multi_class) <= 2

		# assigning targets 
		if label_encoder_binary.transform(['fallacy']) in binary_class:
			target_binary_classification = label_encoder_binary.transform(['fallacy'])
		else:
			target_binary_classification = label_encoder_binary.transform(['non_fallacy'])

		if label_encoder_multi.transform(['none']) in multi_class:
			multi_class.remove(label_encoder_multi.transform(['none']))

		if len(multi_class) == 0:
			target_multi_classification = label_encoder_multi.transform(['none'])
		else:
			target_multi_classification = [multi_class[0]]


		ids = []
		target_binary = []
		target_multi = []

		for i, s in enumerate(text):
			inputs = self.params_b['tokenizer'].encode(s,add_special_tokens=False)
			input_len = len(inputs)
			ids.extend(inputs)
			target_binary.extend([binary[i]] * input_len)
			target_multi.extend([multi[i]] * input_len)


		ids = ids[:self.params_a.max_len - 2]
		target_binary = target_binary[:self.params_a.max_len - 2]
		target_multi = target_multi[:self.params_a.max_len - 2]

		ids = [101] + ids + [102]
		target_binary = [0] + target_binary + [0]
		target_multi = [0] + target_multi + [0]

		mask = [1] * len(ids)
		token_type_ids = [0] * len(ids)

		padding_len = self.params_a.max_len - len(ids)

		ids = ids + ([0] * padding_len)
		mask = mask + ([0] * padding_len)
		token_type_ids = token_type_ids + ([0] * padding_len)
		target_binary = target_binary + ([0] * padding_len)
		target_multi = target_multi + ([0] * padding_len)
		
		return {
			"ids": torch.tensor(ids, dtype=torch.long),
			"mask": torch.tensor(mask, dtype=torch.long),
			"token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
			"target_binary": torch.tensor(target_binary, dtype=torch.long),
			"target_multi": torch.tensor(target_multi, dtype=torch.long),
			"target_binary_classification": torch.tensor(target_binary_classification, dtype=torch.long),
			"target_multi_classification": torch.tensor(target_multi_classification, dtype=torch.long),
			"weights_multi_class": torch.tensor(self.weights_multi_class, dtype=torch.float),
			"weights_binary_class":torch.tensor(self.weights_binary_class, dtype=torch.float),
			"weights_multi_seq":torch.tensor(self.weights_multi_seq, dtype=torch.float),
			"weights_binary_seq":torch.tensor(self.weights_binary_seq, dtype=torch.float)
		}


class FallacyContextDataset:
	def __init__(self, texts,context, binary_span, multi_span, target_binary_classification, target_multi_classification, label_encoder_binary, label_encoder_multi,weights_multi_class, weights_binary_class,weights_multi_seq, weights_binary_seq, params):

		self.texts = texts
		self.binary_span = binary_span
		self.multi_span = multi_span
		self.context = context
		self.target_binary_classification = target_binary_classification
		self.target_multi_classification = target_multi_classification
		self.label_encoder_binary = label_encoder_binary
		self.label_encoder_multi = label_encoder_multi
		self.weights_multi_class = weights_multi_class
		self.weights_binary_class = weights_binary_class
		self.weights_multi_seq = weights_multi_seq
		self.weights_binary_seq = weights_binary_seq
		self.params_a = params[0]
		self.params_b = params[1]


		
	def __len__(self):
		return len(self.texts)


	def __getitem__(self, item):
		text = self.texts[item]
		context = self.context[item]
		binary_span = self.binary_span[item]
		multi_span = self.multi_span[item]
		target_multi_classification = self.target_multi_classification[item]
		target_binary_classification = self.target_binary_classification[item]
		label_encoder_binary = self.label_encoder_binary
		label_encoder_multi = self.label_encoder_multi
		pad_span_multi = self.label_encoder_multi.transform(['none'])
		pad_span_binary = self.label_encoder_binary.transform(['non_fallacy'])

		ids = []
		target_binary = []
		target_multi = []

		for i, s in enumerate(text):
			inputs = self.params_b['tokenizer'].encode(s,add_special_tokens=False)
			input_len = len(inputs)
			ids.extend(inputs)
			target_binary.extend([binary_span[i]] * input_len)
			target_multi.extend([multi_span[i]] * input_len)


		ids = ids[:self.params_a.max_len - 2]
		target_binary = target_binary[:self.params_a.max_len - 2]
		target_multi = target_multi[:self.params_a.max_len - 2]

		ids = [101] + ids + [102]
		target_binary = [0] + target_binary + [0]
		target_multi = [0] + target_multi + [0]

		mask = [1] * len(ids)
		token_type_ids = [0] * len(ids)

		context_encoded = self.params_b['tokenizer'].encode(context,add_special_tokens=False)
		context_encoded = context_encoded[:self.params_a.max_len_context - 1]
		
		context_ids = context_encoded + [102]
		context_mask = [1] * len(context_ids)
		context_token_type_ids = [1] * len(context_ids)
		context_target_binary = list(pad_span_binary) * len(context_encoded) + [0]
		context_target_multi = list(pad_span_multi) * len(context_encoded) + [0]

		padding_len = self.params_a.max_len_context + self.params_a.max_len - len(context_ids) - len(ids)
		final_ids = ids + context_ids + ([0] * padding_len)
		final_mask = mask + context_mask  + ([0] * padding_len)
		final_token_type_ids = token_type_ids + context_token_type_ids + ([0] * padding_len)
		final_target_binary = target_binary + context_target_binary + ([0] * padding_len)
		final_target_multi = target_multi + context_target_multi + ([0] * padding_len)

		return {
			"ids": torch.tensor(final_ids, dtype=torch.long),
			"mask": torch.tensor(final_mask, dtype=torch.long),
			"token_type_ids": torch.tensor(final_token_type_ids, dtype=torch.long),
			"target_binary": torch.tensor(final_target_binary, dtype=torch.long),
			"target_multi": torch.tensor(final_target_multi, dtype=torch.long),
			"target_binary_classification": torch.tensor(target_binary_classification, dtype=torch.long),
			"target_multi_classification": torch.tensor(target_multi_classification, dtype=torch.long),
			"weights_multi_class": torch.tensor(self.weights_multi_class, dtype=torch.float),
			"weights_binary_class":torch.tensor(self.weights_binary_class, dtype=torch.float),
			"weights_multi_seq":torch.tensor(self.weights_multi_seq, dtype=torch.float),
			"weights_binary_seq":torch.tensor(self.weights_binary_seq, dtype=torch.float)
		}