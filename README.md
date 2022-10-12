# superresolution_gan

My simple attempt to write the code for super-resolution gan for satellite imagery based on the paper: Unpaired Image Super-Resolution using Pseudo-Supervision by Shunta Maeda  
The following files are present in the repository:  
**Data Generator**: contains scripts to create training data that can be ingested by the model, feed the location of your data in datagenerator/config.py  
**dataloader**: Contains the dataloader script to transformed data for the model  
**models**: Contains the scripts for generator and discriminator models  
**train config**: contains the hyperparameters required for training  
**utils**: contains helper functions  

