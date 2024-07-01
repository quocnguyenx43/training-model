<!-- * `run_train_cls_task.py`: task 1 & task 2 training <br>
* `run_evaluation_cls_task.py`: task 1 & task 2 evaluation <br>
* `run_train_generation_task.py`: task 3 training <br>
* `run_evaluation_generation_task.py`: task 3 evaluation <br>

# Models using:
	* Task 1 & 2: 
		- XLM-RoBERTa, BERT-base-multilingual-cased, Distilbert-base-multilingual-cased
		- PhoBERT, ViSoBERT, CafeBERT
	* Task 3:
		- ViT5: base
		- BARTpho: word + syllabel -->

<!-- Tạo môi trường:
	conda create --name vuongquoctest2 python=3.10.12
	conda activate vuongquocenv
	conda install [packet] / torchvision
	pip3 install pipreqs
	pip3 install pip-tools
	pipreqs --savepath=requirements.txt && pip-compile

	conda create --name quocenv python=3.10.12
	pip install -r requirements.txt
	conda config --add channels pytorch

	conda env remove --name <environment_name>
	conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

	ls -lha


quy trình chạy:
	+ tạo file .sh
	+ pull
	+ tạo folder: models, results/logs
	+ chmod +x -> .sh -->