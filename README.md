# CS-AF: A Cost-sensitive Multi-classifier Active Fusion Framework for Skin Lesion Classification
Source code of: "CS-AF: A Cost-sensitive Multi-classifier Active Fusion Framework for Skin Lesion Classification"

1. To train the base classifier, just use train.py. 
   The raw image data can be download via the homepage of ISIC 2019:
   https://challenge2019.isic-archive.com/data.html

2. Cost Matrix:
   The 2 demo cost matrices used in our paper are in [].
   
3. The function of the 4 fusion methods (max-voting, average, cs-af and af), sensitivity and specificity are in "ensemble/ensemble_demo.py". 
   Instead of the saved model, we provided the soft labels of our 12 model predictions with 4 distributions, and the corresponding true labels on testing dataset in "ensemble/SF", to provide a more convenient way to reproduce.
   
