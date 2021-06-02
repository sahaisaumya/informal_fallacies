import torch
import transformers
import torch.nn as nn

def loss_fn(output, target, mask, num_labels, weight = None):
	lfn = nn.CrossEntropyLoss()
	active_loss = mask.view(-1) == 1
	active_logits = output.view(-1, num_labels)
	active_labels = torch.where(
		active_loss,
		target.view(-1),
		torch.tensor(lfn.ignore_index).type_as(target)
	)
	return nn.CrossEntropyLoss(weight = weight[0])(active_logits, active_labels)
	loss = lfn(active_logits, active_labels)
	return loss

def loss_fn_classification(outputs, targets, context_flag, weight = None):
	if context_flag:
		return nn.CrossEntropyLoss(weight = weight[0])(outputs, targets)
	return nn.CrossEntropyLoss(weight = weight[0])(outputs, targets.squeeze(1))

class mgnModel(nn.Module):
	def __init__(self, num_binary, num_multi, params):
		super(mgnModel, self).__init__()

		self.params_a = params[0]
		self.params_b = params[1]

		if self.params_a.classes == "binary":
			self.num_classes = num_binary
		if self.params_a.classes == "multi":
			self.num_classes = num_multi
		
		self.bert = transformers.BertModel.from_pretrained(self.params_a.base_model,return_dict=False, output_hidden_states = True)
		self.bert_drop_1 = nn.Dropout(0.3)
		self.bert_drop_pool = nn.Dropout(0.3)
		self.project_dimension = self.num_classes

		self.ff_multi = nn.Linear(768 + self.project_dimension, self.num_classes)
		self.flow_class_seq = nn.Linear(768, self.project_dimension)

		self.ff_2_multi = nn.Linear(768, self.num_classes)
	
	def forward(self, ids, mask, token_type_ids, target_binary, target_multi, target_multi_classification, target_binary_classification,weights_multi_class, weights_binary_class,weights_multi_seq, weights_binary_seq):

		if self.params_a.classes == "binary":
			weight_classification = weights_binary_class
			weight_seq = weights_binary_seq
			target_seq = target_binary
			target_classification = target_binary_classification
		if self.params_a.classes == "multi":
			weight_classification = weights_multi_class
			weight_seq = weights_multi_seq
			target_seq = target_multi
			target_classification = target_multi_classification

		seq_out, pool_out, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

		pool_out = self.bert_drop_pool(pool_out)
		seq_out = self.bert_drop_1(seq_out)
		
		pool_out_cat =  pool_out.unsqueeze(1).repeat(1,self.params_a.max_len,1) 

		if self.params_a.context:
			pool_out_cat =  pool_out.unsqueeze(1).repeat(1,self.params_a.max_len+self.params_a.max_len_context,1)
		pool_out_cat = torch.sigmoid(self.flow_class_seq(pool_out_cat)) 

		bo_multi = torch.cat((pool_out_cat, seq_out), 2) 
		multi = self.ff_multi(bo_multi)
		pool_out = self.ff_2_multi(pool_out) 
		
		loss_multi = loss_fn(multi, target_seq, mask, self.num_classes, weight_seq)
		loss_multi_class = loss_fn_classification(pool_out, target_classification, self.params_a.context ,weight_classification)

		loss = (self.params_a.alpha * loss_multi_class) +( (1-self.params_a.alpha) * loss_multi )  # <- fudge around here
		
		if self.params_a.level == 'token':
			return None , multi, loss

		else:
			return pool_out, None, loss

class bertModel(nn.Module):
	def __init__(self, num_binary, num_multi, params):
		super(bertModel, self).__init__()

		self.params_a = params[0]
		self.params_b = params[1]

		if self.params_a.classes == "binary":
			self.num_classes = num_binary
		if self.params_a.classes == "multi":
			self.num_classes = num_multi
		
		self.bert = transformers.BertModel.from_pretrained(self.params_a.base_model,return_dict=False, output_hidden_states = True)
		self.bert_drop_1 = nn.Dropout(0.3)
		self.bert_drop_2 = nn.Dropout(0.3)

		self.out_seq = nn.Linear(768, self.num_classes)
		self.out_class = nn.Linear(768, self.num_classes)
	
	def forward(self, ids, mask, token_type_ids, target_binary, target_multi, target_multi_classification, target_binary_classification, weights_multi_class, weights_binary_class,weights_multi_seq, weights_binary_seq):

		seq_out, pool_out, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
		if self.params_a.classes == "binary":
			weight_classification = weights_binary_class
			weight_seq = weights_binary_seq
			target_seq = target_binary
			target_classification = target_binary_classification
		if self.params_a.classes == "multi":
			weight_classification = weights_multi_class
			weight_seq = weights_multi_seq
			target_seq = target_multi
			target_classification = target_multi_classification

		seq_out = self.bert_drop_1(seq_out)
		pool_out = self.bert_drop_2(pool_out)

		seq_out = self.out_seq(seq_out)
		pool_out = self.out_class(pool_out)

		loss_seq = loss_fn(seq_out, target_seq, mask, self.num_classes, weight_seq)
		loss_class = loss_fn_classification(pool_out, target_classification, self.params_a.context, weight_classification)

		if self.params_a.level == 'token':
			return None , seq_out, loss_seq

		else:
			return pool_out, None, loss_class





