# AI programming with Python

Please read carefully this file before asking for support.

## Project 1: Used a pre-trained image classifier to identify dog breeds
  
### Context

This first project was mainly intended to make student practice Python and get comfortable with manipulating data structures such as lists or dictionaries.

### Project structure

|_ pet_images

|_ upload_images

|...

|... all program files

### How to run

Command line:

`python check_images.py --dir image_dir --arch nn_arch --dogfile dog_file`
where:
* `image_dir` is the folder containing the images (pet_images or upload_images)
* `nn_arch` is the Neural Network architecture used to classify the images (alexnet, resnet or vgg)
* `dog_file` is the text file containing all breeds of dogs

Example:
`python check_images.py --dir pet_images/ --arch resnet --dogfile dognames.txt`

You can run the classification for all different architectures  in one shot by using:
* run_models_batch.bat (Windows)
* run_models_batch.sh (Linux/Unix)

## Project 2: Created an image classifier using a trained model

### Part 1: develop a code for an image classifier built with Pytorch

#### Context

This part was intended to create an image classifier using a pretrained model.

#### Project structure

|_ assets

|_ flowers

|...

|... all program files

Note: in the project environment, there was an alias on flowers directory (data/flowers/...), just keep in mind that the important thing is having a proper call to the directory, and the directory itself must follow the pattern:
* flowers/train/X : X being a directory X = {1, ..., 102} corresponding to the class id of the flowers (check cat_to_name.json file)
* flowers/valid/X : X being a directory X = {1, ..., 102} corresponding to the class id of the flowers (check cat_to_name.json file)
* flowers/test/X : X being a directory X = {1, ..., 102} corresponding to the class id of the flowers (check cat_to_name.json file)
Of course, within each directory, the images (no specific name is required)

#### How to run

Just run every single cell of the Notebook (.ipynb file) top down.
The cell that can be used for testing is the one containing :
`image_to_test_path = "/data/flowers/test/25/image_06583.jpg"`

`probs, classes = predict(image_to_test_path, model)`

`display_image_results(image_to_test_path)`

Mainly:
* specify an image filename
* predict the classification of that image
* display the results graphically

### Part 2: convert the image classifier into a command line application

#### Context

This part was intended to use our image classifier built previously and make it runnable through command line.

#### Project structure

|_ assets

|_ flowers

|...

|... all program files

Note: in the project environment, there was an alias on flowers directory (data/flowers/...), just keep in mind that the important thing is having a proper call to the directory, and the directory itself must follow the pattern:
* flowers/train/X : X being a directory X = {1, ..., 102} corresponding to the class id of the flowers (check cat_to_name.json file)
* flowers/valid/X : X being a directory X = {1, ..., 102} corresponding to the class id of the flowers (check cat_to_name.json file)
* flowers/test/X : X being a directory X = {1, ..., 102} corresponding to the class id of the flowers (check cat_to_name.json file)
Of course, within each directory, the images (no specific name is required)

#### How to run

This must be run exclusively in command line.

### To train the model:

`python train.py flowers --arch nn_arch --gpu gpuset --save_dir cpt_dir --epochs epoch_num --learning_rate lr --hidden_units hidden_num --dropout drop_num`
where:
* `nn_arch` is the Neural Network architecture used to classify the images (alexnet, resnet18 or vgg19_bn)
* `gpuset` is the usage of GPU for training to speed up the computation (1: yes, 0: no) - please see CUDA note
* `cpt_dir` is the directory for saving the model checkpoints
* `epoch_num` is the number of epochs (iterations) to train the model
* `lr` is the learning rate to train the model
* `hidden_num` is the number of hidden units in our hidden layer
* `drop_num` is the dropout - probability of randomly disactivate somme nodes of our NN (between 0 and 1)

Examples:
* `python train.py flowers --arch alexnet --gpu 1 --save_dir checkpoints --epochs 5 --learning_rate 0.0002 --hidden_units 512 --dropout 0.3`
* `python train.py flowers --arch resnet18 --gpu 1 --save_dir checkpoints --epochs 5 --learning_rate 0.0002 --hidden_units 512 --dropout 0.3`
* `python train.py flowers --arch vgg19_bn --gpu 1 --save_dir checkpoints --epochs 5 --learning_rate 0.0002 --hidden_units 512 --dropout 0.3`

### To test the model:

`python predict.py flower_file cpt_file --arch nn_arch --top_k top_k_num --category_names cat_name --gpu gpuset`
where:
* `flower_file` is the filename of the flower to classify
* `cpt_file` is the model checkpoint file
* `nn_arch` is the Neural Network architecture used to classify the images (alexnet, resnet or vgg)
* `top_k_num` is number of top most likely classes the flower is identified
* `cat_name` is the file containing flower classes ID and their labels
* `gpuset` is the usage of GPU for training to speed up the computation (1: yes, 0: no) - please see CUDA note

Examples:
* `python predict.py flowers/test/25/image_06583.jpg checkpoints/alexnet.pth --arch alexnet --top_k 5 --category_names cat_to_name.json --gpu 1`
* `python predict.py flowers/test/25/image_06583.jpg checkpoints/resnet18.pth --arch resnet18 --top_k 5 --category_names cat_to_name.json --gpu 1`
* `python predict.py flowers/test/25/image_06583.jpg checkpoints/vgg19_bn.pth --arch vgg19_bn --top_k 5 --category_names cat_to_name.json --gpu 1`

Reminder: of course, the checkpoint file must be used accordingly with the right Neural Network architecture, otherwise the Neural Network can't be initialized.

## CUDA

CUDA stands for Compute Device Unified Architecture.
It is a platform that has been created by Nvidia in 2007. The idea is to use GPU processing power to speed up the amount of computations required for today's applications (such as AI based ones).
From what I experienced, GPU computation is at least 10 times faster than CPU one. 

You can check if your GPU is compliant with CUDA by clicking [there](https://developer.nvidia.com/cuda-gpus).

If your GPU is not compatible, then you must run the programs with --gpu 0 or not mention the parameter, as default value is 0.

## Licensing
All that stuff is under MIT licensing:

Copyright (c) 2020 Philippe Leger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
