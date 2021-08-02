import transformers
import argparse

def none_or_str(value):
	if value == 'None':
		return None
	return value

def primary_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--level') #  {"token" "comment"}
	parser.add_argument('--max_len', type=int, default=256)
	parser.add_argument('--max_len_context', type=int, default=64)
	parser.add_argument('--context', type=none_or_str) # {"parent", "title"}
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--valid_batch_size', type=int, default=16)
	parser.add_argument('--test_batch_size', type=int, default=16)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--train_flag', type=int, default=1) # specify 0 to evaluate on test data only
	parser.add_argument('--seed', type=int, default=100)
	parser.add_argument('--base_model', default='bert-base-uncased') 
	parser.add_argument('--model') # {"bertModel" #mgnModel}
	parser.add_argument('--folder') # path to folder with data splits
	parser.add_argument('--classes') #{"multi" "binary"}
	parser.add_argument('--alpha', type=float, default=0.5)
	
	return parser

def secondary_parse(args):
	sec_args = {}
	sec_args['training_file'] = f"../data/{args.folder}/train.txt"
	sec_args['valid_file'] = f"../data/{args.folder}/dev.txt"
	sec_args['test_file'] = f"../data/{args.folder}/test.txt"
	sec_args['tokenizer'] = transformers.BertTokenizer.from_pretrained( args.base_model, do_lower_case=True, local_files_only=True)
	sec_args['model_path'] = f"{args.level}_{args.base_model}_{args.epochs}_{args.model}_{args.folder}_{args.classes}_{args.context}_{args.seed}.bin"
	return sec_args
	
