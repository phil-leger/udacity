#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Imports here
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import time
from collections import OrderedDict
from workspace_utils import active_session

import json

import numpy as np
import pandas as pd
import seaborn as sb

from PIL import Image


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[3]:


# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256), 
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

#print("Data prepation done!")


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[4]:


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[5]:


# TODO: Build and train your network


# In[6]:


# Check if we can use GPU or not - ENABLE GPU whenever necessary!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[7]:


# Select our model. Let's take the best VGG: VGG19bn
model = models.vgg19_bn(pretrained=True)

# Check how it looks like
# model

# Our model looks like:
#(classifier): Sequential(
#    (0): Linear(in_features=25088, out_features=4096, bias=True)
#    (1): ReLU(inplace)
#    (2): Dropout(p=0.5)
#    (3): Linear(in_features=4096, out_features=4096, bias=True)
#    (4): ReLU(inplace)
#    (5): Dropout(p=0.5)
#    (6): Linear(in_features=4096, out_features=1000, bias=True)
#  )


# In[8]:


# Now, we need to change the classifier to fit with what we need

# No gradient
for param in model.parameters():
    param.requires_grad = False

# Create the new classifier
n_input_units = 25088 # Magic: this is 224 x 224 (the image size) / 2
n_hidden_layer_1_units = 4096 #4096
n_hidden_layer_2_units = 2048 #2048
#n_hidden_layer_3_units = 1024
#n_hidden_layer_4_units = 512
n_output_units = 102 # number of flower classes
dropout = 0.2
learn_rate = 0.00025

# Define the classifier
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(n_input_units, n_hidden_layer_1_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=dropout)),    
                          ('fc2', nn.Linear(n_hidden_layer_1_units, n_hidden_layer_2_units)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(p=dropout)), 
#                          ('fc3', nn.Linear(n_hidden_layer_2_units, n_hidden_layer_3_units)),
#                          ('relu3', nn.ReLU()),
#                          ('dropout3', nn.Dropout(p=0.2)),     
#                          ('fc4', nn.Linear(n_hidden_layer_3_units, n_hidden_layer_4_units)),
#                          ('relu4', nn.ReLU()),
#                          ('dropout4', nn.Dropout(p=0.2)),
#                          ('fc5', nn.Linear(n_hidden_layer_4_units, n_output_units)),
                          ('fc5', nn.Linear(n_hidden_layer_2_units, n_output_units)), 
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

# Instantiate the classifier
model.classifier = classifier


# In[9]:


# Define the loss function (as we use logsoftmax, we use NLLLoss)
criterion = nn.NLLLoss()
# For information: if we dont use the logsoftmax for activation of output, we can use the crossentropy loss function

# Define the optimizer. REMINDER: only train the classifier parameters, feature parameters must be frozen!
# REMARK: play with learning rate? We use Adam, as suggested by Mat's lessons
optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

# We switch the model to the GPU if available
model.to(device)

# NEW MODEL:
#  (classifier): Sequential(
#    (fc1): Linear(in_features=25088, out_features=4096, bias=True)
#    (relu1): ReLU()
#    (dropout1): Dropout(p=0.2)
#    (fc2): Linear(in_features=4096, out_features=2048, bias=True)
#    (relu2): ReLU()
#    (dropout2): Dropout(p=0.2)
#    (fc3): Linear(in_features=2048, out_features=1024, bias=True)
#    (relu3): ReLU()
#    (dropout3): Dropout(p=0.2)
#    (fc4): Linear(in_features=1024, out_features=512, bias=True)
#    (relu4): ReLU()
#    (dropout4): Dropout(p=0.2)
#    (fc5): Linear(in_features=512, out_features=102, bias=True)
#    (output): LogSoftmax()
#  )


# In[10]:


# Time to train and validate the NN!

# Initialization
epochs = 5 # let's start safe!
steps = 0
valid_every_x_steps = 5
running_loss = 0

# Save the losses in lists for training and validation purposes
train_losses, valid_losses = [], []


# In[11]:


# Main Loop
for e in range(epochs):
    
    for images, labels in train_loader:

        # For each print_every_steps we will do a validation pass
        steps += 1
        
        # Move input and label tensors to the default device!
        images, labels = images.to(device), labels.to(device)        

        # Initialize gradients
        optimizer.zero_grad()
        
        # Classic NN processing steps
        ouputs = model(images)
        loss = criterion(ouputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    if steps % valid_every_x_steps == 0:

        # Every X steps, we do a validation pass
        valid_loss = 0
        accuracy = 0
        
        # Turn off gradients
        with torch.no_grad():

            # Disable the dropout during validation & testing and whenever we make predictions
            model.eval()
            
            # Validation pass here
            for images, labels in valid_loader:

                # Move input and label tensors to the default device
                images, labels = images.to(device), labels.to(device) 
                
                # We run our NN with a feed forward
                outputs = model(images)
                valid_loss += criterion (outputs, labels)
                ps = torch.exp(outputs)
                
                # We take each class having highest probability
                top_p, top_class = ps.topk(1, dim=1)
                
                # We compare this class with the real label to check if it's correct or not
                equals = top_class == labels.view(*top_class.shape)
                
                # We calculate the accuracy
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        model.train()
        
        train_losses.append(running_loss/len(train_loader))
        valid_losses.append(valid_loss/len(valid_loader)) 

        print("Epoch: {}/{}" .format(e+1, epochs),
              "Training loss: {:.3f}" . format(train_losses[e]),
              "Validation loss: {:.3f}" . format(valid_losses[e]),
              "Validation Accuracy: {:.3f}". format(accuracy/len(valid_loader)))


# In[12]:


# Display results in a plot
plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.legend(frameon=False)


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[13]:


# TODO: Do validation on the test set

epochs = 1
test_loss = 0
accuracy = 0

# Save the losses for test purpose
test_losses = []

for e in range(epochs):

    # Turn off gradients
    with torch.no_grad():

        # Disable the dropout during testing
        model.eval()

        # Validation pass here
        for images, labels in test_loader:

            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device) 

            # We run our NN with a feed forward
            outputs = model(images)
            test_loss += criterion (outputs, labels)
            ps = torch.exp(outputs)

            # We take each class having highest probability
            top_p, top_class = ps.topk(1, dim=1)

            # We compare this class with the real label to check if it's correct or not
            equals = top_class == labels.view(*top_class.shape)

            # We calculate the accuracy
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    test_losses.append(test_loss/len(test_loader)) 

    print("Epoch: {}/{}" .format(e+1, epochs),
          "Test loss: {:.3f}" . format(test_losses[e]),
          "Test Accuracy: {:.3f}". format(accuracy/len(test_loader)))


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[14]:


print(model.state_dict)


# In[15]:


# TODO: Save the checkpoint
# Means we freeze all parameters: 
# model=XXXXX, outputsize, statedict, optimizer statedict, hyperparams (epochs, lr, etc.), class_to_idx, 
# etc. hard_code the classifier
def save_checkpoint():

#    model.class_to_idx = image_datasets['train'].class_to_idx
    model.class_to_idx = train_dataset.class_to_idx
    try:
        checkpoint = { 'arch': 'vgg19_bn',
                       'model': model.classifier,
                       'input_size': n_input_units,
                       'output_size': n_output_units,
                       'hidden_layer1_size': n_hidden_layer_1_units,
                       'hidden_layer2_size': n_hidden_layer_2_units,
#                       'hidden_layers': [each.out_features for each in model.hidden_layers],
                       'state_dict': model.state_dict(),
                       'optimizer_dict': optimizer.state_dict(),
                       'class_to_idx': model.class_to_idx,
                       'learn_rate': learn_rate,
                       'dropout' : dropout,
                       'epochs': epochs}

        torch.save(checkpoint, 'checkpoint.pth')
        print("Checkpoint saved with success!")

    except Exception as ex:
        print("An exception occured : {}" . format(ex))


# In[16]:


#print(model.parameters)


# In[17]:


save_checkpoint()


# In[18]:


# Check hat the file has been created
get_ipython().run_line_magic('ls', '-la | grep checkpoint.pth')


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[19]:


# TODO: Write a function that loads a checkpoint and rebuilds the model # CHECK WITH ALEX... DOESNT SEEM TO WORK...
def load_checkpoint(filepath):

    try:
        checkpoint = torch.load(filepath)

        if checkpoint['arch'] == 'vgg19_bn':
            model = models.vgg19_bn(pretrained = True)
        
        # Rebuild it as it is
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer1_size'])),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=checkpoint['dropout'])),    
                          ('fc2', nn.Linear(checkpoint['hidden_layer1_size'], checkpoint['hidden_layer2_size'])),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(p=checkpoint['dropout'])), 
#                          ('fc3', nn.Linear(n_hidden_layer_2_units, n_hidden_layer_3_units)),
#                          ('relu3', nn.ReLU()),
#                          ('dropout3', nn.Dropout(p=0.2)),     
#                          ('fc4', nn.Linear(n_hidden_layer_3_units, n_hidden_layer_4_units)),
#                          ('relu4', nn.ReLU()),
#                          ('dropout4', nn.Dropout(p=0.2)),
#                          ('fc5', nn.Linear(n_hidden_layer_4_units, n_output_units)),
                          ('fc5', nn.Linear(checkpoint['hidden_layer2_size'], checkpoint['output_size'])), 
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

        model.classifier = classifier
        
        model.load_state_dict(checkpoint['state_dict'])
        # Below: useful if I still want to improve my model
        optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learn_rate'])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])

        #load the classes
        model.class_to_idx = checkpoint['class_to_idx']

        ### START: DOESN'T SEEM NECESSARY
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()
        ### END: DOESN'T SEEM NECESSARY
        
        print("Checkpoint loaded with success!")
        
    except Exception as ex:
        print("An exception occured : {}" . format(ex))
    else:    
        return model


# In[20]:


# Load test: warning, we must first run the import library and optimizer initializaiton
model = load_checkpoint('checkpoint.pth')


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[21]:


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    max_size = 256
    crop_size = 16
    width = image.width
    height = image.height

    # Resize the image
    if width > height:
        image = image.resize((int(width * max_size / height), max_size))
    else:
        image = image.resize((max_size, int(height * max_size / width)))

    # Crop the image
    image = image.crop((crop_size, crop_size, max_size - crop_size, max_size - crop_size))
    
    # Image normalization
    np_image = np.array(image)
    # Array of 3 dimensions: rows, columns and colour channel (3))
    # input[channel] = (input[channel] - mean[channel]) / std[channel]

    np_image_mean = np_image.mean(axis = 0) # average value of each column
    np_image_std = np_image.std(axis = 0) # std value of each column    
    np_image_norm = (np_image - np_image_mean) / np_image_std

    # As the shape is (rows, columns, color_channels) and pytorch needs (color_channels, rows, columns) we need to transpose it
    return np_image_norm.transpose(2, 0, 1)


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[22]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    # Phil ADD: we want the name of the flower as a title!
    fig.suptitle(title)
    
    ax.imshow(image)
    
    return ax


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[43]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file

    # First, load the image - can be opened by im.show()
    im = Image.open(image_path)  

    # Process the image - then the image will become a numpy array
    img = process_image(im)
    
    # Convert the img numpy to tensor
    img_tensor = torch.from_numpy(img)

    # FIX RuntimeError: expected stride to be a single integer value or a list of 1 values to match the convolution dimensions, but got stride=[1, 1]
    # To confirm with alex: we do this because we do not run the NN on n images but only 1
    # Meaning, it is expecting 4 dim, while only 3 are provided : with unsqueeze, we add a 4th dimension in 1st position
    img_tensor.unsqueeze_(0)

    # FIX tensor type error: we convert data into float
    img_tensor = img_tensor.float()
    
    # Get the class probabilities
    ps = torch.exp(model(img_tensor))

    top_p, top_class = ps.topk(topk, dim=1)

    # We have tensors (dim 0, 1) and want to return numpy data
    np_top_p = top_p[0].numpy()
    np_top_class = top_class[0].numpy()
    
    # Watchout: the top_class is returning a class_to_idx, not the id of the class to get the label!!!
    key_list = list(model.class_to_idx.keys())
    val_list = list(model.class_to_idx.values())

    for i in range(len(np_top_class)):
        np_top_class[i] = key_list[val_list.index(np_top_class[i])]
    
    # We return numpy data
    return np_top_p, np_top_class


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[48]:


########################################################################
# TODO: Display an image along with the top 5 classes
########################################################################

def display_image_results(image_path):
    # Transform the top n class indexes into class labels LIST.
    flower_labels = []
    flower_probs = []

    for i in range(len(classes)):
        flower_labels.append(cat_to_name[str(classes[i])])
        flower_probs.append(float(probs[i]))

    # Put the data in a dataframe
    df = pd.DataFrame(flower_labels, columns = ['flower_labels'])
    df['flower_probs'] = pd.DataFrame(flower_probs)

    # Display the photo and the 5 classes below
    #plt.figure(figsize=[2.33, 4.66])  # 224 x 448 px

    #plt.figure()
    #plt.subplot(2,1,1)

    ### ASK ALEX. IT WORKS BUT IT IS NOT VERY NICE ###
    #plt.imshow(torch.from_numpy(process_image(Image.open(image_to_test_path)))) #  TOFIX!!! Invalid
    plt.show(imshow(torch.from_numpy(process_image(Image.open(image_path))),None,flower_labels[0])) 
    #plt.show(imshow(torch.from_numpy(img)))

    # plt.subplot(2,1,2)
    # 1 qualitative + 1 quantitative = seaborn barplot !
    base_color = sb.color_palette()[0]
    sb.barplot(data = df, x = 'flower_probs', y = 'flower_labels', color = base_color)


# In[49]:


#########
# Tests for 1 image...
# Test (text)
image_to_test_path = "/data/flowers/test/25/image_06583.jpg"
probs, classes = predict(image_to_test_path, model)
display_image_results(image_to_test_path)
#print(probs)
#print(classes)
#print(classes[1])


# In[25]:


########################################################################
# TO REVIEWER: BELOW IS BUNCH OF CODES USED FOR DEBUGGING AND TESTING...
########################################################################


# In[39]:


# Print label of one id
print(cat_to_name['20'])


# In[52]:


print(classes.shape)
print(classes[0][4])
print(len(classes[0]))


# In[2]:


# Some bash command lines
get_ipython().run_line_magic('ls', '-la')
get_ipython().run_line_magic('pwd', '')
# pwd: '/home/workspace/aipnd-project'
# flowers is our directory for images


# In[69]:


#% cd flowers
#% cd test
#print(np.random.randint(1,102))
get_ipython().run_line_magic('cd', 'data')


# In[156]:


#########
# Tests for debugging... processing image
im = Image.open("/data/flowers/test/25/image_06583.jpg")  
# plt.imshow(im)

# Im, stored in a np array is (rows, columns, color channel) 715, 500, 3

# let's debug the process code...
img = process_image(im) # Img returns a transposed (color channel, rows, columns): 3, 715, 500 GOOD! 

print("Shape", img.shape) 
imshow(torch.from_numpy(img))   # IT WORKS !!!


# In[34]:


#########
# Tests for debugging... predict code [inside]...

# let's debug the 
img_tensor = torch.from_numpy(img)

print("img_tensor", img_tensor.shape)
print("img_tensor", type(img_tensor))
# Add a 4th dimension as we process only one image
img_tensor.unsqueeze_(0)
print("img_tensor unsqueezed", img_tensor.shape)
print("img_tensor unsqueezed", type(img_tensor))

# Let's try...
img_tensor = img_tensor.float()

# BUGGED HERE!!! problem of dimensions...
ps = torch.exp(model(img_tensor))


# In[12]:


print(len(test_loader))


# In[10]:


get_ipython().run_line_magic('ls', '-la')


# In[9]:


#%rm checkpoint.pth


# In[38]:


# The flower class value returns an ID !!! not the class!
print(model.class_to_idx)
print(cat_to_name['25'])

# print(list(cat_to_name.keys())[list(cat_to_name.values()).index(25)])
key_list = list(model.class_to_idx.keys())
val_list = list(model.class_to_idx.values())

print(key_list[val_list.index(25)])

# to do for whole vector
for i in range(len(classes)):
    print (classes[i]) 
    print (key_list[val_list.index(classes[i])])
    print (cat_to_name[key_list[val_list.index(classes[i])]])


# In[ ]:




