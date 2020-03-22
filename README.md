# CS-AF
Source code of: "CS-AF: A Cost-sensitive Multi-classifier Active Fusion Framework for Skin Lesion Classification"

1. To train the base classifier, just use train.py. The split of train/validation/test are in "data/txt_file"
   The raw image data can be download via the homepage of ISIC 2019:
   https://challenge2019.isic-archive.com/data.html

2. For ensemble. The 4 ensemble functions (max-voting, average, cs-af and af) are in "ensemble/ensemble_demo.py". 
   Instead of the saved model, we provide the 10 soft label and true label on testing dataset in "ensemble/SF" to give a more convenient way to implement.
   
