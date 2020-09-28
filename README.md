# CS-AF: A Cost-sensitive Multi-classifier Active Fusion Framework for Skin Lesion Classification
Source code of: "CS-AF: A Cost-sensitive Multi-classifier Active Fusion Framework for Skin Lesion Classification"

# 1. Train classifiers:

   To train the base classifier, what you need are:
   
      train.py: training process.
      
      Model.py: define different models.
      
      get_data.py: get image data by defining dataset and data loader.
      
   The raw image data can be download via the homepage of ISIC 2019:
   
   https://challenge2019.isic-archive.com/data.html
   
   
   1.1 Put all the images in same folder;
   
   1.2 The train/val/test split files are in "/data/txt_file";
   
   1.3 Change the path of image folder and .txt file in "get_data.py" before training.
   
   Example command in the terminal:
   
      python train.py --gpu 0 --network inceptionv3 --epochs 50 --train_batchsz 32 --num_class 8 --lr 1e-3
   
# 2. Quick demo for CS-AF:

   We provide the demo code to run our CS-AF on 48 (12 x 4) models and plot the curves of accuracy and total cost.

   The 2 demo cost matrices used in our paper are in "ensemble/cost_matrix".
   
   The function of the 4 fusion methods (max-voting, average, cs-af and af) are in "ensemble/ensemble_demo.py". 
   
   Instead of the saved model, we provide the soft labels of our 12 model predictions with 4 distributions (in "/ensemble/SF"), the corresponding true labels also included. 
   
   To produce the curve of accuracy and total cost, just download everything and run "/ensemble/fusion_demo.py". 
   
   The number of random repetitions can be modified at the line 125 of "fusion_demo.py".
   
