rem !/bin/sh
rem  */AIPND-revision/intropyproject-classify-pet-images/run_models_batch.bat
rem                                                                              
rem  PROGRAMMER: Philippe LEGER
rem  DATE CREATED: 02/07/2020                                  
rem  REVISED DATE: 
rem  PURPOSE: Runs all three models to test which provides 'best' solution.
rem           Please note output from each run has been piped into a text file.
rem 
rem  Usage: run_models_batch.bat    -- will run program from commandline within Project Workspace
rem   
python check_images.py --dir pet_images/ --arch resnet  --dogfile dognames.txt > pet-images_resnet.txt
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt > pet-images_alexnet.txt
python check_images.py --dir pet_images/ --arch vgg  --dogfile dognames.txt > pet-images_vgg.txt
