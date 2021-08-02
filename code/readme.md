
Use following to train model and evaluate on test data. All the argument options can be found in `config.py`. 

`python train.py --level $LEVEL --context $CONTEXT --model $MODEL --classes $CLASSES --folder $FOLDER`

`level` : task specification, ie token level or comment level <br>
`context` : specify if title or parent is to be added as context; remove the parameter for the no context case <br>
`model` : specify if BERT or multigranularity model is to be used <br>
`classes` : specify if output will be binary (fallacy exists or not) or multi class (the exact fallacy type) <br>
`folder` : path to data splits
