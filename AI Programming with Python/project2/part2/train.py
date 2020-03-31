# ImageClassifier/train.py
#
# PROGRAMMER: Philippe LEGER
# DATE CREATED: 20200328@1920                                
# REVISED DATE: 20200329@
# PURPOSE: train file used for training/validating a new model
#

# Import required modules
from time import time, sleep

# Import personal modules
from phil_lib import get_input_args
from phil_lib import check_input_args
from phil_lib import init_data
from phil_lib import set_device
from phil_models import select_model
from phil_models import create_classifier
from phil_models import run_classifier
from phil_models import save_checkpoint

########################################################################
def main():

    # Initialize timer
    start_time = time()

    # Get input arguments 
    in_arg = get_input_args("train")

    # Manage arguments
    check_input_args(in_arg, "train")

    # Initialize a model_param dict used throughout the main function
    model_param = {}
    
    # Initialize data
    train_loader, valid_loader, model_param = init_data(in_arg, model_param, "train")

    # Use CUDA power or slow CPU. Reminder: we activate GPU only if CUDA is available AND in_arg.gpu = 1
    model_param['device'] = set_device(in_arg)
    
    # Set the NN model
    model, model_param = select_model(model_param, in_arg)

    print("Model chosen:{}".format(model))
    
    # Create the classifier
    model, model_param = create_classifier(model, model_param, in_arg)

    # DEBUG PRINTS START
    print("Model modified:{}".format(model))
    print("Architecture:{}".format(model_param['arch']))    
    print("Criterion:{}".format(model_param['criterion']))
    print("Optimizer:{}".format(model_param['optimizer']))
    print("LearnRate:{}".format(model_param['learning_rate']))
    print("Dropout:{}".format(model_param['dropout']))
    print("Epochs:{}".format(model_param['epochs']))
    print("Class_to_dix:{}".format(model_param['class_to_idx']))
    print("Save_dir:{}".format(model_param['save_dir']))
    print("Data_dir:{}".format(model_param['data_dir']))
    print("Device:{}".format(model_param['device']))
    # DEBUG PRINTS FINISH    
    
    # Train the model - ENABLE GPU!
    run_classifier(model, model_param, train_loader, valid_loader)

    # Save the model
    save_checkpoint(model, model_param)
    
    # It's finished!
    end_time = time()
    
    # Computes overall runtime and prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )    
    
    # END OF PROGRAM, THANK YOU FOR READING!
    
# Call to main function to run the program
if __name__ == "__main__":
    main()