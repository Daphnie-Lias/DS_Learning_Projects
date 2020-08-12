# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

To test train.py :

This implementation accepts 2 pre trained models (vgg16 : default and alexnet)
Sample Input:
    python train.py data_dir --arch ‘alexnet' --learning_rate 0.01 --hidden_units 512 --epochs 3
   	
To test predict.py
Sample Input:
	python predict.py --image "flowers/train/13/image_05765.jpg"  --top_k 3