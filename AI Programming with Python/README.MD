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
python check_images.py --dir image_dir --arch nn_arch
where:
* image_dir is the folder containing the images (pet_images or upload_images)
* nn_arch is the Neural Network architecture used to classify the images (alexnet, resnet or vgg)

Example:
python check_images.py --dir pet_images/ --arch resnet

You can run the classification for all different architectures  in one shot by using:
* run_models_batch.bat (Windows)
* run_models_batch.sh (Linux/Unix)

## Project 2: Created an image classifier using a trained model

### Part 1: develop a code for an image classifier built with Pytorch

#### Context

This part was intended to create an image classifier using a pretrained model.

#### Project structure

|_ pet_images
|_ upload_images
|...
|... all program files

#### How to run

Just run every single cell of the Notebook (.ipynb file) top down.
The cell that can be used for testing is the one containing :
`image_to_test_path = "/data/flowers/test/25/image_06583.jpg"
probs, classes = predict(image_to_test_path, model)
display_image_results(image_to_test_path)`

Mainly:
* specify an image filename
* predict the classification of that image
* display the results graphically

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
