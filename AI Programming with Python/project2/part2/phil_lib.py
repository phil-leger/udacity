# ImageClassifier/phil_lib.py
#                                                                             
# PROGRAMMER: Philippe LEGER
# DATE CREATED: 20200328@1912                                  
# REVISED DATE: 20200329@12xx
# PURPOSE: Package of libraries/functions used by main programs train.py and predict.py
#


# Import required libraries
import argparse
import sys
import os
import torch
import json

from torchvision import datasets, transforms, models
from os import path
from PIL import Image
import numpy as np

########################################################################
def get_input_args(phase="train"):
    """
    Retrieves and parses command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to create and define command line arguments. If 
    the user fails to provide some arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments for train:
      1. Image Folder as data_dir (positional argument)
      2. Checkpoint Folder as --save_dir with default value ''      
      3. CNN Model Architecture as --arch with default value 'vgg19_bn'
      4. Learning Rate --learning_rate with default value 0.00025
      5. Epochs as --epochs with default value 5
      6. Hidden units as --hidden_units with default value 512
      7. GPU as --GPU to activate GPU mode or not with default value 0
      8. Batch size as --batch_size with default value 32
      9. Dropout as --dropout with default value 0.2 
      
    Command Line Arguments for predict:
      1. Image file as image_file (positional argument)
      2. Checkpoint file as checkpoint (positional argument)
      3. Top K classes to display for image classification as --top_k with default value 5
      4. Category Names as --cat_to_name with default value "cat_to_name.json"
      5. GPU as --GPU to activate GPU mode or not with default value 0
      6. CNN Model Architecture as --arch with default value 'vgg19_bn'

    This function returns these arguments as an ArgumentParser object.
    Parameters:
     phase - from which program phase are we requiring arguments
    Returns:
     parse_args() - data structure (namespace) that stores the command line arguments object  
    """
    # Create parser using ArgumentParser !!! To improve to manage dynamically only one list of parser[phase] like a dict ({argparse.ArgumentParser()})!
    parser = argparse.ArgumentParser()
    
    # Create command line arguments
    if phase == "train":
        parser.add_argument("data_dir", type = str, help="Image Folder where flower images are located (path starts from program path).")
        parser.add_argument("--save_dir", type = str, default = "checkpoints", help="Directory where checkpoints are saved. Default: current path")
        parser.add_argument("--arch", type = str, default = "vgg19_bn", help="CNN Model Architecture: alexnet, resnet18 or vgg19_bn. Default: vgg19_bn")
        parser.add_argument("--learning_rate", type = float, default = 0.00025, help="Neural Network learning rate. Default: 0.00025")
        parser.add_argument("--epochs", type = int, default = 5, help="Neural Network number of epochs. Default: 5")
        parser.add_argument("--hidden_units", type = int, default = 512, help="Neural Network hidden units. Default: 512")
        parser.add_argument("--gpu", type = int, default = 0, help="Use GPU power to train the model: 0 (no) or 1 (yes). Your graphical card must be compatible with CUDA. Check online at https://developer.nvidia.com/cuda-gpus! Default: 0")
        # extra !
        parser.add_argument("--batch_size", type = int, default = 32, help="Images batch size to train/test the model. Default: 32")
        parser.add_argument("--dropout", type = float, default = 0.2, help="Dropout percentage to apply during training. Default: 0.2") 

    else:
        # phase == "predict"
        parser.add_argument("image_file", type = str, help="Relative path of flower image including filename (path starts from program path).")
        parser.add_argument("checkpoint", type = str, help="Relative path of checkpoint of model (path starts from program path).")
        parser.add_argument("--top_k", type = int, default = 5, help="K classes to display for image classification. Default: 5")
        parser.add_argument("--category_names", type = str, default = "cat_to_name.json", help="Filename containing classes and their labels. Default: cat_to_name.json") 
        parser.add_argument("--gpu", type = int, default = 0, help="Use GPU power to infer the model: 0 (no) or 1 (yes). Your graphical card must be compatible with CUDA. Check online at https://developer.nvidia.com/cuda-gpus! Default: 0")
        # extra !
        parser.add_argument("--arch", type = str, default = "vgg19_bn", help="CNN Model Architecture: alexnet, resnet18 or vgg19_bn. Default: vgg19_bn")        
        
    return parser.parse_args()


########################################################################
def check_input_args(in_arg, phase="train"):
    """
    Controls command line arguments provided by the user when
    they run the program from a terminal window. 
    Parameters:
     input_args - command line arguments
     phase - from which program phase are we requiring arguments

    Returns:
     none  
    """

    if phase=="train": 
        # Check that flowers directory exists 
        if not path.isdir(in_arg.data_dir):
            print("For data loading: can't find directory '{}' starting from '{}'. Please check the paths and run again!" . format(in_arg.data_dir, os.getcwd()))
            sys.exit(0)
    
        # Check that checkpoints directory exists
        if not path.isdir(in_arg.save_dir):
            print("For checkpoints saving: can't find directory '{}' starting from '{}'. Please check the paths and run again!" . format(in_arg.save_dir, os.getcwd()))
            sys.exit(0)         
        
    else:
        # phase == predict
        # Check that the flower name exists. Example: "/data/flowers/test/25/image_06583.jpg"
        if not path.isfile(in_arg.image_file):
            print("Image file: can't find file '{}' starting from '{}'. Please check the path, filename and run again!" . format(in_arg.image_file, os.getcwd()))
            sys.exit(0)             
        
        if not path.isfile(in_arg.checkpoint):
            print("Checkpoint file: can't find file '{}' starting from '{}'. Please check the path, filename and run again!" . format(in_arg.checkpoint, os.getcwd()))
            sys.exit(0)
            
        if in_arg.category_names and not path.isfile(in_arg.category_names):
            print("Category names file: can't find file '{}' starting from '{}'. Please check the path, filename and run again!" . format(in_arg.category_names, os.getcwd()))
            sys.exit(0)  
        
    # All cases

    # Check that the architecture is supported
    if in_arg.arch not in ['alexnet', 'resnet18', 'vgg19_bn']:
        print("Architecture can only be: alexnet, resnet18 or vgg19_bn. Please check the architecture and run again!")
        sys.exit(0)    
    
    # Check that a valid value has been set for gpu
    if in_arg.gpu != 0 and in_arg.gpu != 1:
        print("GPU can only be set to 0 (disable) or 1 (enable)! Please check the value and run again!")
        sys.exit(0)
        
        
########################################################################
def init_data(in_arg, model_param, phase="train"):
    """
    Initialise the transformations to be performed on dataset 
    Parameters:
     in_arg - command line arguments
     model_param - dictionary of our model parameters     
     phase - from which program phase are we requiring arguments

    Returns:
     train_loader - loader of training data
     valid_loader - loader of validation data
     model_param - dictionary of our model parameters     
    """       
    # Firstly, set the directories
    # PRE-REQUISITES: 
    # train & valid sets (1 per folder) must exist within the in_arg.data_dir (to improve if I have some time later on)
    # train folder must be "train", validation folwer must be "valid"
    # each file must be correctly classified (=within the correct id folder). file name doesn't matter
    model_param['data_dir'] = in_arg.data_dir
    train_dir = model_param['data_dir'] + '/train'
    valid_dir = model_param['data_dir'] + '/valid'

    model_param['save_dir'] = in_arg.save_dir
    
    # Prepare the transformations for train & validation sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    try:
        # Load the datasets with ImageFolder
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

        model_param['class_to_idx'] = train_dataset.class_to_idx
        
        # TODO: Using the image datasets and the trainforms, define the dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=in_arg.batch_size, shuffle = True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=in_arg.batch_size, shuffle = True)

        # Initialize the cat_to_name catalog
        #with open(in_arg.cat_to_name, 'r') as f:
            #cat_to_name = json.load(f)
        #    model_param['cat_to_name'] = json.load(f)

    except Exception as e:
        print("An exception occured: {}.".format(e))
        sys.exit(0)

    print("Data loading completed!")

    # Return all parameters we will need later on
    return train_loader, valid_loader, model_param
    
    
########################################################################
def set_device(in_arg):
    """
    Activates the CUDA power if prompted AND available
    Parameters:
     in_arg - command line arguments

    Returns:
     "cuda" if CUDA is enabled, "cpu" if CUDA is disabled  
    """       
    
    return torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu == 1 else "cpu")
    
    
########################################################################
def load_category_names(category_names_file):
    """
    Load the category file names of flowers
    Parameters:
     category_names_file - path name (starting from app current working directory) including the filename and its extension
    Returns:
     catalog of flower classes and labels
    """    

    with open(category_names_file, 'r') as f:
        return json.load(f)
    
    
########################################################################
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    """
    Process the image (scaling, croping and normalizing a PIL image for a PyTorch model.
    Parameters:
     image - PIL image
    Returns:
     Numpy array transformation of the original PIL image
    """  
    
    # Some parameters initialization
    max_size = 256
    crop_size = 16
    width = image.width
    height = image.height

    # Resize the image
    if width > height:
        image = image.resize((int(width * max_size / height), max_size))
    else:
        image = image.resize((max_size, int(height * max_size / width)))

    # Crop the image so that we get max_size - 2 * crop_size px. In our case 224 x 224 px
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


########################################################################

def predict(model, model_param, topk=5):
    """
    Predict the flower name with the topk most likely classes using a trained deep learning model.
    Parameters:
     image_path - path name of the flower image
     model - our NN model
     topk - the K most probable classes
    Returns:
     probability and id of the top k most likely classes
    """  
   
    # First, load the image
    im = Image.open(model_param['image_file'])  

    # Process the image - then the image will become a numpy array
    img = process_image(im)
    
    # Convert the img numpy to tensor
    img_tensor = torch.from_numpy(img)

    # FIX RuntimeError: expected stride to be a single integer value or a list of 1 values to match the convolution dimensions, but got stride=[1, 1]
    # Meaning, it is expecting 4 dim, while only 3 are provided : with unsqueeze, we add a 4th dimension in 1st position
    img_tensor.unsqueeze_(0)

    # FIX tensor type error: we convert data into float
    img_tensor = img_tensor.float()

    # Fix on RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'weight': move data to device
    img_tensor = img_tensor.to(model_param['device'])
    
    # Get the class probabilities
    ps = torch.exp(model(img_tensor))

    # Get the topk classes results
    top_p, top_class = ps.topk(topk, dim=1)

    # We have tensors (dim 0, 1) and want to return numpy data
    # Fix the TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. Instead of X[0].numpy()...
    np_top_p = top_p[0].cpu().clone().numpy()
    np_top_class = top_class[0].cpu().clone().numpy()
    lst_classes = [] # we use a list for the flowers labels
    
    # Watchout: the top_class is returning a class_to_idx, not the id of the class to get the label!!!
    key_list = list(model.class_to_idx.keys())
    val_list = list(model.class_to_idx.values())

    for i in range(len(np_top_class)):
        np_top_class[i] = key_list[val_list.index(np_top_class[i])]
        lst_classes.append(display_label(np_top_class[i], model_param['category_names']))
        
    # We return numpy data
    return np_top_p, np_top_class, lst_classes


########################################################################
def display_label(f_class, catalog):
    """
    Predict the flower name with the topk most likely classes using a trained deep learning model.
    Parameters:
     image_path - path name of the flower image
     model - our NN model
     topk - the K most probable classes
    Returns:
     probability and id of the top k most likely classes
    """  
    # Transform the top n class indexes into class labels LIST.
    return catalog[str(f_class)]
