<!-- # Models using:
	* Task 1 & 2: 
		- XLM-RoBERTa, BERT-base-multilingual-cased, Distilbert-base-multilingual-cased
		- PhoBERT, ViSoBERT, CafeBERT
	* Task 3:
		- ViT5: base
		- BARTpho: word + syllabe

# Tạo môi trường:
	conda create --name envnam python=3.10.12
	conda activate envnam
	conda install ...
	pip3 install pipreqs
	pip3 install pip-tools
	pipreqs --savepath=requirements.txt && pip-compile

	conda create --name envname python=3.10.12
	pip install -r requirements.txt

# Quy trình chạy:
	+ Create .sh file
	+ Git pull
	+ Create folder models/, results/logs
	+ chmod +x
	+ run .sh file -->