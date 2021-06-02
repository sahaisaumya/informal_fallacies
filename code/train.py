import pandas as pd
import numpy as np

import joblib
import torch
import ast
import argparse

def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn

from sklearn import preprocessing
from sklearn import model_selection
from sklearn.utils.class_weight import compute_class_weight

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import utils
import model as all_models

parser = config.primary_parse()
params_a = parser.parse_args()
params_b = config.secondary_parse(params_a)

torch.manual_seed(params_a.seed)
np.random.seed(params_a.seed)

if __name__ == "__main__":

	print(f"Running {params_a.model} at {params_a.level} level, using data at {params_a.folder} for {params_a.classes} class case, context = {params_a.context} len  = {params_a.max_len_context}, seed is {params_a.seed}")
	print("Save path is ", params_b['model_path'])
	train_flag = params_a.train_flag
	
	if params_a.context:

		sentences_train, sentences_context_train, binary_span_train, multi_span_train, target_binary_classification_train, target_multi_classification_train, enc_binary_span, enc_multi_span, weights_multi_class, weights_binary_class, weights_multi_seq, weights_binary_seq = utils.process_data_context(params_b['training_file'], [params_a, params_b])
		sentences_val, sentences_context_val, binary_span_val, multi_span_val, target_binary_classification_val, target_multi_classification_val,_,_, _,_ ,_,_= utils.process_data_context(params_b['valid_file'], [params_a, params_b])
		sentences_test, sentences_context_test, binary_span_test, multi_span_test, target_binary_classification_test, target_multi_classification_test,_,_,_,_,_,_ = utils.process_data_context(params_b['test_file'], [params_a, params_b])

		train_dataset = dataset.FallacyContextDataset(texts=sentences_train, context=sentences_context_train, binary_span=binary_span_train, multi_span=multi_span_train, target_binary_classification=target_binary_classification_train, target_multi_classification=target_multi_classification_train, label_encoder_binary=enc_binary_span, label_encoder_multi=enc_multi_span, weights_multi_class=weights_multi_class, weights_binary_class=weights_binary_class, weights_multi_seq=weights_multi_seq, weights_binary_seq=weights_binary_seq, params = [params_a, params_b])
		test_dataset = dataset.FallacyContextDataset(texts=sentences_test, context=sentences_context_test, binary_span=binary_span_test, multi_span=multi_span_test, target_binary_classification=target_binary_classification_test, target_multi_classification=target_multi_classification_test, label_encoder_binary=enc_binary_span, label_encoder_multi=enc_multi_span,weights_multi_class=weights_multi_class, weights_binary_class=weights_binary_class,weights_multi_seq=weights_multi_seq, weights_binary_seq=weights_binary_seq, params = [params_a, params_b])
		val_dataset = dataset.FallacyContextDataset(texts=sentences_val, context=sentences_context_val, binary_span=binary_span_val, multi_span=multi_span_val, target_binary_classification=target_binary_classification_val, target_multi_classification=target_multi_classification_val, label_encoder_binary=enc_binary_span, label_encoder_multi=enc_multi_span, weights_multi_class=weights_multi_class, weights_binary_class=weights_binary_class,weights_multi_seq=weights_multi_seq, weights_binary_seq=weights_binary_seq, params = [params_a, params_b])

	else:
		sentences_train, binary_span_train, multi_span_train, enc_binary_span, enc_multi_span, weights_multi_class, weights_binary_class, weights_multi_seq, weights_binary_seq = utils.process_data_no_context(params_b['training_file'], [params_a, params_b])
		sentences_val, binary_span_val, multi_span_val, _,_, _,_ ,_,_= utils.process_data_no_context(params_b['valid_file'], [params_a, params_b])
		sentences_test, binary_span_test, multi_span_test, _,_,_,_,_,_ = utils.process_data_no_context(params_b['test_file'], [params_a, params_b])

		train_dataset = dataset.FallacyDataset(texts=sentences_train, binary=binary_span_train, multi=multi_span_train, label_encoder_binary=enc_binary_span, label_encoder_multi=enc_multi_span, weights_multi_class=weights_multi_class, weights_binary_class=weights_binary_class, weights_multi_seq=weights_multi_seq, weights_binary_seq=weights_binary_seq, params = [params_a, params_b])
		test_dataset = dataset.FallacyDataset(texts=sentences_test, binary=binary_span_test, multi=multi_span_test, label_encoder_binary=enc_binary_span, label_encoder_multi=enc_multi_span,weights_multi_class=weights_multi_class, weights_binary_class=weights_binary_class,weights_multi_seq=weights_multi_seq, weights_binary_seq=weights_binary_seq, params = [params_a, params_b])
		val_dataset = dataset.FallacyDataset(texts=sentences_val, binary=binary_span_val, multi=multi_span_val, label_encoder_binary=enc_binary_span, label_encoder_multi=enc_multi_span, weights_multi_class=weights_multi_class, weights_binary_class=weights_binary_class,weights_multi_seq=weights_multi_seq, weights_binary_seq=weights_binary_seq, params = [params_a, params_b])


	num_binary = len(list(enc_binary_span.classes_))
	num_multi = len(list(enc_multi_span.classes_))
	target_multi = enc_multi_span.inverse_transform(np.arange(num_multi))
	target_binary = enc_binary_span.inverse_transform(np.arange(num_binary))

	train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params_a.train_batch_size, num_workers=4, shuffle=True)
	test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params_a.test_batch_size, num_workers=4)
	val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params_a.valid_batch_size, num_workers=4)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if train_flag:
		model = getattr(all_models, params_a.model)(num_binary, num_multi, [params_a, params_b])
		model.to(device)

		
		num_train_steps = int(len(sentences_train) / params_a.train_batch_size * params_a.epochs)
		optimizer = AdamW(model.parameters(), lr=5e-5)
		scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

		best_loss = np.inf
		best_f1 = 0
		for epoch in range(params_a.epochs):
			train_loss = utils.train_fn(train_data_loader, model, optimizer, device, scheduler)
			val_loss, val_macrof1_multi_seq, val_macrof1_multi_class = utils.eval_fn(val_data_loader, model, device, [params_a, params_b])
			if params_a.level == "token":
				val_macrof1_multi_class = 0
			else:
				val_macrof1_multi_seq = 0
			print(f"Epoch = {epoch+1} Train Loss = {train_loss} Valid Loss = {val_loss} Val macroF1 token = {val_macrof1_multi_seq} Val macroF1 class = {val_macrof1_multi_class}")
			
			if best_f1 < ((val_macrof1_multi_seq + val_macrof1_multi_class)):
				print("saving model")
				torch.save(model.state_dict(), params_b['model_path'])
				best_f1 = ((val_macrof1_multi_seq + val_macrof1_multi_class) )


	print("Running on test set")
	b_model = getattr(all_models, params_a.model)(num_binary, num_multi, [params_a, params_b])

	print("Loading best model")
	b_model.load_state_dict(torch.load(params_b['model_path']))
	b_model.to(device)

	test_loss, report_multi_seq, report_multi_class, cm = utils.test_fn(test_data_loader, b_model, device, target_multi, target_binary, [params_a, params_b])

	print(f"Test Loss = {test_loss}")
	print("report_multi_seq ",report_multi_seq)
	print("report_multi_class ",report_multi_class)
	print(cm)