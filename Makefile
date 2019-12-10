.PHONY model:
model:
	nohup python train_model.py > train-$1.log

