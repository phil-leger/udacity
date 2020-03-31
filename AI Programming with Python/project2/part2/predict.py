# ImageClassifier/predict.py
#
# PROGRAMMER: Philippe LEGER
# DATE CREATED: 20200329@2228                                
# REVISED DATE: 20200330@1904
# PURPOSE: predict file used for predicting the flower name from an image
#
# Import required modules
from time import time, sleep

# Import personal modules
from phil_lib import get_input_args
from phil_lib import check_input_args
from phil_lib import set_device
from phil_lib import load_category_names
from phil_lib import predict
from phil_models import load_checkpoint


########################################################################
def main():

    # Initialize timer
    start_time = time()
    
    # Get input arguments 
    in_arg = get_input_args("predict")

    # Manage arguments
    check_input_args(in_arg, "predict")    

    # Initialize a model_param dict used throughout the main function
    model_param = {}
    
    # Load the category names file
    model_param['category_names'] = load_category_names(in_arg.category_names) # To access: model_param['category_names']['82'] :)

    # Use CUDA power or slow CPU. 
    # QUESTION FOR REVIEWER: it seems that, if the checkpoint has CUDA activated, same configuration must be applied while loading the checkpoint ?
    model_param['device'] = set_device(in_arg)
    
    # Load the checkpoint
    model, model_param = load_checkpoint(in_arg.checkpoint, model_param)

    # Predict the image!
    model_param['image_file'] = in_arg.image_file
    probs, classes, classes_labels = predict(model, model_param, in_arg.top_k)

    """ DEBUG PRINTS START
    print("Model modified:{}".format(model))
    print("Architecture:{}".format(model_param['arch']))    
    print("Criterion:{}".format(model_param['criterion']))
    print("Optimizer:{}".format(model_param['optimizer']))
    print("LearnRate:{}".format(model_param['learning_rate']))
    print("Dropout:{}".format(model_param['dropout']))
    print("Epochs:{}".format(model_param['epochs']))
    #print("Class_to_dix:{}".format(model_param['class_to_idx']))
    print("Save_dir:{}".format(model_param['save_dir']))
    print("Data_dir:{}".format(model_param['data_dir']))
    print("Device:{}".format(model_param['device']))
    """
    # DEBUG PRINTS FINISH
    
    # It's finished!
    end_time = time()
    
    # Computes overall runtime and prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )       

    # Display results of the prediction in a more sexy way :)
    print("The image '{}' is:".format(model_param['image_file']))
    for i in range(len(probs)):
        print("A {} with {:.2%} probability".format(classes_labels[i] ,probs[i]))
    
    # END OF PROGRAM, THANK YOU FOR READING!
    
# Call to main function to run the program
if __name__ == "__main__":
    main()