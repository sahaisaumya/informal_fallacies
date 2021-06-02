import numpy
import ast
import torch

from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn import preprocessing


def process_data_context(path, params):
	with open(path) as f:
		content = f.readlines()
	
	enc_binary_span = preprocessing.LabelEncoder()
	enc_multi_span = preprocessing.LabelEncoder()

	sentences = []
	binary_span = []
	multi_span = []
	sent_lengths = []
	sentences_context = []

	for line in content:
		all_parts = line.split('\t')
		sentences.append(ast.literal_eval(all_parts[1]))

		if params[0].context == "parent":
			con = all_parts[5]
		if params[0].context == "title":
			con = all_parts[6]
		if params[0].context == "both":
			con = all_parts[6] + " " + all_parts[5]
		sentences_context.append(con)

		binary_span.append(ast.literal_eval(all_parts[3]))
		multi_span.append(ast.literal_eval(all_parts[4]))
	enc_binary_span.fit([item for sublist in binary_span for item in sublist])
	enc_multi_span.fit([item for sublist in multi_span for item in sublist])

	binary_span = [enc_binary_span.transform(x) for x in binary_span]
	multi_span = [enc_multi_span.transform(x) for x in multi_span]
	
	target_multi_classification = []
	target_binary_classification = []
	
	for i in range(len(sentences)):
		binary_class = list(set(binary_span[i]))
		multi_class = list(set(multi_span[i]))
		if enc_binary_span.transform(['fallacy']) in binary_class:
			target_binary_classification.append(enc_binary_span.transform(['fallacy'])[0])
		else:
			target_binary_classification.append(enc_binary_span.transform(['non_fallacy'])[0])

		if enc_multi_span.transform(['none']) in multi_class:
			multi_class.remove(enc_multi_span.transform(['none']))

		if len(multi_class) == 0:
			target_multi_classification.append(enc_multi_span.transform(['none'])[0])
		else:
			target_multi_classification.append(multi_class[0])

	class_weights_multi_class = compute_class_weight('balanced', numpy.unique(target_multi_classification), target_multi_classification)
	class_weights_binary_class = compute_class_weight('balanced', numpy.unique(target_binary_classification), target_binary_classification)

	pad_span_multi = enc_multi_span.transform(['none'])
	pad_span_binary = enc_binary_span.transform(['non_fallacy'])
	binary_span_class_count = []
	multi_span_class_count = []
	for item in range(len(sentences)):
		#### for context
		context_encoded = params[1]['tokenizer'].encode(sentences_context[item],add_special_tokens=False)
		context_encoded = context_encoded[:params[0].max_len_context - 1]
		binary_span_class_count.extend(list(pad_span_binary) * len(context_encoded))
		multi_span_class_count.extend(list(pad_span_multi) * len(context_encoded))
		#### for text
		for i, s in enumerate(sentences[item]):
			inputs = params[1]['tokenizer'].encode(s,add_special_tokens=False)
			input_len = len(inputs)
			binary_span_class_count.extend([binary_span[item][i]] * input_len)
			multi_span_class_count.extend([multi_span[item][i]] * input_len)
	class_weights_multi_seq = compute_class_weight('balanced', numpy.unique(multi_span_class_count) , multi_span_class_count)
	class_weights_binary_seq = compute_class_weight('balanced', numpy.unique(binary_span_class_count), binary_span_class_count)
	
	return sentences, sentences_context, binary_span, multi_span, target_binary_classification, target_multi_classification, enc_binary_span, enc_multi_span, class_weights_multi_class, class_weights_binary_class, class_weights_multi_seq, class_weights_binary_seq

def process_data_no_context(path, params):
	with open(path) as f:
		content = f.readlines()
	
	enc_binary_span = preprocessing.LabelEncoder()
	enc_multi_span = preprocessing.LabelEncoder()

	sentences = []
	binary_span = []
	multi_span = []

	for line in content:
		all_parts = line.split('\t')
		sentences.append(ast.literal_eval(all_parts[1]))
		binary_span.append(ast.literal_eval(all_parts[3]))
		multi_span.append(ast.literal_eval(all_parts[4]))

	enc_binary_span.fit([item for sublist in binary_span for item in sublist])
	enc_multi_span.fit([item for sublist in multi_span for item in sublist])

	binary_span = [enc_binary_span.transform(x) for x in binary_span]
	multi_span = [enc_multi_span.transform(x) for x in multi_span]
	
	target_multi_classification = []
	target_binary_classification = []
	for i in range(len(sentences)):
		binary_class = list(set(binary_span[i]))
		multi_class = list(set(multi_span[i]))
		if enc_binary_span.transform(['fallacy']) in binary_class:
			target_binary_classification.append(enc_binary_span.transform(['fallacy'])[0])
		else:
			target_binary_classification.append(enc_binary_span.transform(['non_fallacy'])[0])

		if enc_multi_span.transform(['none']) in multi_class:
			multi_class.remove(enc_multi_span.transform(['none']))

		if len(multi_class) == 0:
			target_multi_classification.append(enc_multi_span.transform(['none'])[0])
		else:
			target_multi_classification.append(multi_class[0])

	class_weights_multi_class = compute_class_weight('balanced', numpy.unique(target_multi_classification), target_multi_classification)
	class_weights_binary_class = compute_class_weight('balanced', numpy.unique(target_binary_classification), target_binary_classification)

	pad_span_multi = enc_multi_span.transform(['none'])
	pad_span_binary = enc_binary_span.transform(['non_fallacy'])
	binary_span_class_count = []
	multi_span_class_count = []
	for item in range(len(sentences)):
		for i, s in enumerate(sentences[item]):
			inputs = params[1]['tokenizer'].encode(s,add_special_tokens=False)
			input_len = len(inputs)
			binary_span_class_count.extend([binary_span[item][i]] * input_len)
			multi_span_class_count.extend([multi_span[item][i]] * input_len)
	class_weights_multi_seq = compute_class_weight('balanced', numpy.unique(multi_span_class_count) , multi_span_class_count)
	class_weights_binary_seq = compute_class_weight('balanced', numpy.unique(binary_span_class_count), binary_span_class_count)

	return sentences, binary_span, multi_span, enc_binary_span, enc_multi_span, class_weights_multi_class, class_weights_binary_class, class_weights_multi_seq, class_weights_binary_seq

def perf_fn(output, target, mask, token_type_ids):
	output = output.argmax(2)
	return  target.cpu().numpy(), output.cpu().numpy(), mask.cpu().numpy(), token_type_ids.cpu().numpy()

def train_fn(dataldr, model, optimizer, device, scheduler):
	model.train()
	final_loss = 0
	for data in tqdm(dataldr, total = len(dataldr)):
		for k,v in data.items():
			data[k] = v.to(device)
		optimizer.zero_grad()
		_,_, loss = model(**data)
		loss.backward()
		optimizer.step()
		scheduler.step()
		final_loss += loss.item()
	return final_loss / len(dataldr)

def eval_fn(dataldr, model, device, params):
	if params[0].classes == "binary":
		target_seq = 'target_binary'
		target_classification = 'target_binary_classification'
	if params[0].classes == "multi":
		target_seq = 'target_multi'
		target_classification = 'target_multi_classification'

	model.eval()
	final_loss = 0

	y_true_multi_seq = []
	y_pred_multi_seq = []

	y_true_multi_class = []
	y_pred_multi_class = []

	macro_f1_multi_seq = 0
	macro_f1_multi_class = 0

	mask_filt = []
	token_type_filt = []
	for data in tqdm(dataldr, total = len(dataldr)):
		for k, v in data.items():
			data[k] = v.to(device)
		multi_class, multi_seq, loss = model(**data)
		final_loss += loss.item()

		if multi_seq is not None:
			y_t_multi_seq, y_p_multi_seq, msk, tti = perf_fn(multi_seq, data[target_seq], data['mask'], data['token_type_ids'])
			y_true_multi_seq.extend(y_t_multi_seq)
			y_pred_multi_seq.extend(y_p_multi_seq)
			mask_filt.extend(msk)
			token_type_filt.extend(tti)
		if multi_class is not None:
			if params[0].context:
				y_true_multi_class.extend(data[target_classification].cpu().numpy())
				y_pred_multi_class.extend(multi_class.argmax(1).detach().cpu().numpy())
			else:
				y_true_multi_class.extend(data[target_classification].squeeze(1).cpu().numpy())
				y_pred_multi_class.extend(multi_class.argmax(1).detach().cpu().numpy())

	if multi_seq is not None:
		y_true_multi_seq = numpy.concatenate(y_true_multi_seq).ravel()
		y_pred_multi_seq = numpy.concatenate(y_pred_multi_seq).ravel()
		mask_filt = numpy.concatenate(mask_filt).ravel()
		token_type_filt = numpy.concatenate(token_type_filt).ravel()
		condition = mask_filt ==1
		if params[0].context:
			condition = (mask_filt ==1) & (token_type_filt == 0)
		macro_f1_multi_seq = f1_score(numpy.extract(condition, y_true_multi_seq), numpy.extract(condition, y_pred_multi_seq), average='macro')
	if multi_class is not None:
		macro_f1_multi_class = f1_score(y_true_multi_class, y_pred_multi_class, average = 'macro')
	return final_loss / len(dataldr), macro_f1_multi_seq, macro_f1_multi_class

def test_fn(dataldr, model, device, target_cls_multi, target_cls_binary, params):
	if params[0].classes == "binary":
		target_cls = target_cls_binary
		target_seq = 'target_binary'
		target_classification = 'target_binary_classification'
	if params[0].classes == "multi":
		target_cls = target_cls_multi
		target_seq = 'target_multi'
		target_classification = 'target_multi_classification'

	model.eval()
	final_loss = 0

	y_true_multi_seq = []
	y_pred_multi_seq = []

	y_true_multi_class = []
	y_pred_multi_class = []

	report_multi_seq = "Not reported"
	report_multi_class = "Not reported"

	mask_filt = []
	token_type_filt = []
	for data in tqdm(dataldr, total = len(dataldr)):
		for k, v in data.items():
			data[k] = v.to(device)

		multi_class, multi_seq, loss = model(**data)
		final_loss += loss.item()
		
		if multi_seq is not None:
			y_t_multi_seq, y_p_multi_seq, msk, tti = perf_fn(multi_seq, data[target_seq], data['mask'], data['token_type_ids'])
			y_true_multi_seq.extend(y_t_multi_seq)
			y_pred_multi_seq.extend(y_p_multi_seq)
			mask_filt.extend(msk)
			token_type_filt.extend(tti)

		if multi_class is not None:
			if params[0].context:
				y_true_multi_class.extend(data[target_classification].cpu().numpy())
				y_pred_multi_class.extend(multi_class.argmax(1).detach().cpu().numpy())
			else:
				y_true_multi_class.extend(data[target_classification].squeeze(1).cpu().numpy())
				y_pred_multi_class.extend(multi_class.argmax(1).detach().cpu().numpy())

	if multi_seq is not None:
		y_true_multi_seq = numpy.concatenate(y_true_multi_seq).ravel()
		y_pred_multi_seq = numpy.concatenate(y_pred_multi_seq).ravel()
		mask_filt = numpy.concatenate(mask_filt).ravel()
		token_type_filt = numpy.concatenate(token_type_filt).ravel()
		condition = mask_filt ==1
		if params[0].context:
			condition = (mask_filt ==1) & (token_type_filt == 0)
		
		report_multi_seq = classification_report(numpy.extract(condition, y_true_multi_seq), numpy.extract(condition, y_pred_multi_seq), digits=4, target_names = target_cls)
	
	
	if multi_class is not None:
		report_multi_class = classification_report(y_true_multi_class, y_pred_multi_class, digits=4, target_names = target_cls)
		cm = confusion_matrix(y_true_multi_class, y_pred_multi_class)
		print(target_cls)
		print(cm)

	return final_loss / len(dataldr), report_multi_seq, report_multi_class, None
