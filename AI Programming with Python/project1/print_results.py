#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/print_results.py
#                                                                             
# PROGRAMMER: Philippe LEGER
# DATE CREATED: 20200203
# REVISED DATE: 
# PURPOSE: Create a function print_results that prints the results statistics
#          from the results statistics dictionary (results_stats_dic). It 
#          should also allow the user to be able to print out cases of misclassified
#          dogs and cases of misclassified breeds of dog using the Results 
#          dictionary (results_dic).  
#         This function inputs:
#            -The results dictionary as results_dic within print_results 
#             function and results for the function call within main.
#            -The results statistics dictionary as results_stats_dic within 
#             print_results function and results_stats for the function call within main.
#            -The CNN model architecture as model wihtin print_results function
#             and in_arg.arch for the function call within main. 
#            -Prints Incorrectly Classified Dogs as print_incorrect_dogs within
#             print_results function and set as either boolean value True or 
#             False in the function call within main (defaults to False)
#            -Prints Incorrectly Classified Breeds as print_incorrect_breed within
#             print_results function and set as either boolean value True or 
#             False in the function call within main (defaults to False)
#         This function does not output anything other than printing a summary
#         of the final results.
##
# TODO 6: Define print_results function below, specifically replace the None
#       below by the function definition of the print_results function. 
#       Notice that this function doesn't to return anything because it  
#       prints a summary of the results using results_dic and results_stats_dic
# 
def print_results(results_dic, results_stats_dic, model, 
                  print_incorrect_dogs = False, print_incorrect_breed = False):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats_dic - Dictionary that contains the results statistics (either
                   a  percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - Indicates which CNN model architecture will be used by the 
              classifier function to classify the pet images,
              values must be either: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """    
    # print the model that has been used
    print("Results Summary for CNN model '{}':".format(model))
                
    # Now we need to print some numbers (Z - number of images, B - number of dog images, D - number of non-dog images)
    print("Number of images: {}".format(results_stats_dic['n_images']))
    print("Number of dog images: {}".format(results_stats_dic['n_dogs_img']))
    print("Number of non-dog images: {}".format(results_stats_dic['n_notdogs_img']))

    # Now we need to print some pct (1a - percentage of correctly classified dogs, 2 - percentage of correctly classified dog breeds, 1b - percentage of correctly classified NON-dogs, opt - percentage of correct matches)
    for f in results_stats_dic.keys():
        if f[0:3] == "pct":
            print("Percentage {}: {}%".format(f,results_stats_dic[f]))
    
    # Print misclassifications of dogs
    if print_incorrect_dogs:
        print("Printing misclassified dogs...")
        # Check if we have misclassified dogs
        if results_stats_dic['n_correct_dogs'] + results_stats_dic['n_correct_notdogs'] != results_stats_dic['n_images']:
            # Loop all the results to get the ones with misclassified labels 
            print("Here they are:")
            for k in results_dic.keys():
                # Check if EITHER pet label is 'a dog' and classifier 'not a dog' OR pet label is 'not a dog' and classifier 'a dog'
                if sum(results_dic[k][3:]) == 1:
                    print("Pet image: '{}' - pet label '{}' - classifier label '{}'".format(k, results_dic[k][0], results_dic[k][1]))
        else:
            print("No misclassified dog.")
    else:
        print("User decided not to print misclassified dogs.")
                    
    # Print misclassifications of dogs breeds
    if print_incorrect_breed:
        print("Printing misclassified dogs breeds...")
        # Check if we have misclassified dogs breeds
        if results_stats_dic['n_correct_dogs']  != results_stats_dic['n_correct_breed']:
            # Loop all the results to get the ones with misclassified breed labels
            print("Here they are:")
            for k in results_dic.keys():
                # Check if pet label and classifier 'is a dog' AND no match between pet label and classifier label
                if sum(results_dic[k][3:]) == 2 and results_dic[k][2] == 0:
                    print("Pet image: '{}' - pet label '{}' - classifier label '{}'".format(k, results_dic[k][0], results_dic[k][1]))
        else:
            print("No misclassified dogs breed.")
    else:
        print("User decided not to print misclassified dogs breeds.")