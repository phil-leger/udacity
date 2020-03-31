# ImageClassifier/phil_models.py
#
# PROGRAMMER: Philippe LEGER
# DATE CREATED: 20200329@1257                                
# REVISED DATE: 20200330@1731
# PURPOSE: models file used for generating/managing models
#

import sys

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

########################################################################
def select_model(model_param, in_arg):
    """
    Selects the NN model based on command line argument "arch"
    Parameters:
     model_param - dictionary of our model parameters
     in_arg - command line arguments

    Returns:
     Selected NN model (object)
     model_param - dictionary of our model parameters
    """  

    # We will manage 3 choices: alexnet, resnet18, vgg19_bn
    alexnet = models.alexnet(pretrained=True)
    resnet18 = models.resnet18(pretrained=True)
    vgg19_bn = models.vgg19_bn(pretrained=True)
    
    models_dict = {'alexnet': alexnet, 'resnet18': resnet18, 'vgg19_bn': vgg19_bn}
    
    if in_arg.arch not in models_dict.keys():
        print("CNN model '{}' is not managed. It can only be chosen among alexnet, resnet18 and vgg19_bn. Try again!".format(in_arg.arch))
        sys.exit(0)

    model_param['arch'] = in_arg.arch
        
    return models_dict[in_arg.arch], model_param

########################################################################
def create_classifier(model, model_param, in_arg):
    """
    Create the classifier based on command line argument "arch" and chosen model
    Parameters:
     model - our CNN model
     model_param - dictionary of our model parameters
     in_arg - command line arguments

    Returns:
     model - our CNN model
     model_param - dictionary of our model parameters
    """ 
    
    # No gradient
    for param in model.parameters():
        param.requires_grad = False
        
    # Create the new classifier
   
    # We need to "hard code" a classifier per architecture
    models_input_units = {'alexnet': 9216, 'resnet18': 512, 'vgg19_bn': 25088} # Output for the final layer 25088 = 512 x 7 x 7
    
    model_param['n_input_units'] = models_input_units[model_param['arch']]  
    model_param['n_hidden_layer_units'] = in_arg.hidden_units
    model_param['n_output_units'] = 102 # number of flower classes
    model_param['dropout'] = in_arg.dropout
    model_param['learning_rate'] = in_arg.learning_rate
    
    # Define the classifier
    if model_param['arch'] in ['alexnet', 'vgg19_bn']: 
        
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(model_param['n_input_units'], model_param['n_hidden_layer_units'])),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=model_param['dropout'])),    
                          #('fc2', nn.Linear(n_hidden_layer_1_units, n_hidden_layer_2_units)),
                          #('relu2', nn.ReLU()),
                          #('dropout2', nn.Dropout(p=dropout)), 
                          ('fc2', nn.Linear(model_param['n_hidden_layer_units'], model_param['n_output_units'])), 
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

        # Instantiate the last linear layer which is classifier
        model.classifier = classifier
        
        # Define the optimizer. REMINDER: only train the classifier parameters, feature parameters must be frozen!
        model_param['optimizer'] = optim.Adam(model.classifier.parameters(), lr=model_param['learning_rate'])
        
    else:
        # As 3 choice only are managed, here we have the resnet18 case only 

        fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(model_param['n_input_units'], model_param['n_hidden_layer_units'])),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=model_param['dropout'])),    
            #('fc2', nn.Linear(n_hidden_layer_1_units, n_hidden_layer_2_units)),
            #('relu2', nn.ReLU()),
            #('dropout2', nn.Dropout(p=dropout)), 
            ('fc2', nn.Linear(model_param['n_hidden_layer_units'], model_param['n_output_units'])), 
            ('output', nn.LogSoftmax(dim=1))
        ]))

        # Instantiate the last linear layer which is fc
        model.fc = fc
        
        # Define the optimizer. REMINDER: only train the classifier parameters, feature parameters must be frozen!
        model_param['optimizer'] = optim.Adam(model.fc.parameters(), lr=model_param['learning_rate'])
   
    # Define the loss function (as we use logsoftmax, we use NLLLoss)
    # criterion = nn.NLLLoss()
    model_param['criterion'] = nn.NLLLoss()
    
    # We switch the model to the GPU if available
    model.to(model_param['device'])
    
    # Get additional parameters
    model_param['epochs'] = in_arg.epochs
    
    return model, model_param
    
########################################################################
def run_classifier(model, model_param, train_loader, valid_loader):
    """
    Create the classifier based on command line argument "arch" and chosen model
    Parameters:
     model - our CNN model
     model_param - dictionary of our model parameters
    Returns:
     None  
    """
    # Initialization
    epochs = model_param['epochs']
    steps = 0
    valid_every_x_steps = 5
    running_loss = 0

    # Save the losses in lists for training and validation purposes
    train_losses, valid_losses = [], []
    
    # Main Loop
    for e in range(epochs):

        for images, labels in train_loader:

            # For each print_every_steps we will do a validation pass
            steps += 1

            # Move input and label tensors to the default device!
            # images, labels = images.to(device), labels.to(device)        
            images, labels = images.to(model_param['device']), labels.to(model_param['device'])
            
            # Initialize gradients
            #optimizer.zero_grad()
            model_param['optimizer'].zero_grad()

            # Classic NN processing steps
            ouputs = model(images)
            #loss = criterion(ouputs, labels)
            loss = model_param['criterion'](ouputs, labels)
            loss.backward()
            #optimizer.step()
            model_param['optimizer'].step()

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
                    # images, labels = images.to(device), labels.to(device) 
                    images, labels = images.to(model_param['device']), labels.to(model_param['device'])

                    # We run our NN with a feed forward
                    outputs = model(images)
                    #valid_loss += criterion (outputs, labels)
                    valid_loss += model_param['criterion'](outputs, labels)
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


      
            
########################################################################
def save_checkpoint(model, model_param):
    """
    Save a checkpoint for our model
    Parameters:
     model - our CNN model
     model_param - dictionary of our model parameters
    Returns:
     None  
    """

#    model.class_to_idx = image_datasets['train'].class_to_idx
    #model.class_to_idx = train_dataset.class_to_idx
    try:
        checkpoint = { 'arch': model_param['arch'],
                       #'model': model.classifier,
                       'model': model.classifier if model_param['arch'] in ['alexnet', 'vgg19_bn'] else model.fc,
                       'data_dir': model_param['data_dir'],
                       'input_size': model_param['n_input_units'],
                       'output_size': model_param['n_output_units'],
                       'hidden_layer_size': model_param['n_hidden_layer_units'],
                       'state_dict': model.state_dict(),
                       'optimizer_dict': model_param['optimizer'].state_dict(),
                       'class_to_idx': model_param['class_to_idx'], #model.class_to_idx,
                       'learning_rate': model_param['learning_rate'],
                       'dropout': model_param['dropout'],
                       'cpt_file_name': model_param['arch']+'.pth',
                       'epochs': model_param['epochs']}

        torch.save(checkpoint, model_param['save_dir'] + '/' + checkpoint['cpt_file_name'])
        print("Checkpoint saved with success!")

    except Exception as e:
        print("An exception occured : {}" . format(e))
        print("Impossible to save checkpoint !")
        sys.exit(0)
        
########################################################################
def load_checkpoint(checkpoint_file, model_param):
    """
    Load a checkpoint for our model
    Parameters:
     checkpoint_file - path name (starting from app current working directory) including the filename and its extension
     model_param - dictionary of our model parameters
    Returns:
     model - our model
     model_param - dictionary of our model parameters
    """
    
    try:
        checkpoint = torch.load(checkpoint_file)

        # Create the right model
        if checkpoint['arch'] == 'alexnet':
            model = models.alexnet(pretrained=True)
        elif checkpoint['arch'] == 'resnet18':
            model = models.resnet18(pretrained=True)
        else:
            # Can only be vgg19_bn
            model = models.vgg19_bn(pretrained=True)

        # Define the classifier
        if checkpoint['arch'] in ['alexnet', 'vgg19_bn']: 

            # Instantiate the last linear layer which is classifier
            model.classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer_size'])),
                ('relu1', nn.ReLU()),
                ('dropout1', nn.Dropout(p=checkpoint['dropout'])),    
                ('fc2', nn.Linear(checkpoint['hidden_layer_size'], checkpoint['output_size'])), 
                ('output', nn.LogSoftmax(dim=1))
            ]))

            # Define the optimizer. REMINDER: only train the classifier parameters, feature parameters must be frozen!
            model_param['optimizer'] = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])

        else:
            # As 3 choices only are managed, here we have the resnet18 case only 

            # Instantiate the last linear layer which is fc            
            model.fc = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer_size'])),
                ('relu1', nn.ReLU()),
                ('dropout1', nn.Dropout(p=checkpoint['dropout'])),    
                ('fc2', nn.Linear(checkpoint['hidden_layer_size'], checkpoint['output_size'])), 
                ('output', nn.LogSoftmax(dim=1))
            ]))

            # Define the optimizer. REMINDER: only train the classifier parameters, feature parameters must be frozen!
            model_param['optimizer'] = optim.Adam(model.fc.parameters(), lr=checkpoint['learning_rate'])


        model.load_state_dict(checkpoint['state_dict'])
        # Below: useful if I still want to improve my model
        model_param['optimizer'].load_state_dict(checkpoint['optimizer_dict'])

        # Define the loss function (as we use logsoftmax, we use NLLLoss)
        model_param['criterion'] = nn.NLLLoss()

        # We switch the model to the GPU if CUDA is available and in_arg.gpu = 1
        model.to(model_param['device'])
    
        # Get additional parameters
        model_param['epochs'] = checkpoint['epochs']
    
        # Load the classes/indexes of flowers
        model.class_to_idx = checkpoint['class_to_idx']

        ### START: DOESN'T SEEM NECESSARY
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()
        ### END: DOESN'T SEEM NECESSARY
        
        print("Checkpoint loaded with success!")
        
    except Exception as e:
        print("An exception occured : {}" . format(e))
    else:    
        return model, model_param



########################################################################