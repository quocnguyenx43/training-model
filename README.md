train máy trường sử dụng anydesk

Tạo môi trường local:
	conda create --name vuongquoc python=3.10.12 <br>
	conda activate vuongquoc <br>
	conda install [packet] / torchvision <br>
	pip3 install pipreqs <br>
	pip3 install pip-tools <br>
	pipreqs --savepath=requirements.txt && pip-compile <br>

Máy ảo:
	conda create --name vuongquoc python=3.10.12 <br>
	pip install -r requirements.txt <br>
	conda env remove --name <environment_name> <br>