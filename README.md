# Code to create and optimize CNN models for glaucoma stage classification using two imaging modalities, fundus images and 3D OCT volumes.

![alt text](https://github.com/MarcosMF86/IEEE_GlaucomaGradingUsingMultimodalImaging/blob/main/ProjectImages/visual%20abstract.png?raw=true)

Resources and extra documentation for the manuscript "Glaucoma Grading Using Multimodal Imaging and Multilevel CNN". This repository contains source codes used for grading glaucoma stage using fundus images and OCT volume scans, relesead by GAMMA challenge

# Problem
Automate glaucoma grading stage using two images modalities, fundus images and 3D OCT volumes scan.

# Proposed Method
We proposed a glaucoma grading method usind fundus images and 3D OCT volumes through deep learning. In first step, a hyperparameter optimization was carried out, evaluating different CNN models, handling each imaging modality as input, whole fundus images, optic disc region image cropped from fundus images and 3D OCT. In second step, ensemble and feature combination strategies were evaluated to grade glaucoma stage.
# Dataset
In the official challenge website: <a href="https://aistudio.baidu.com/aistudio/competition/detail/807/0/introduction" target="_blank">GAMMA</a>, the dataset can be found, and classifications results can be submitted. 
# Instructions
1. Download the dataset from the official GAMMA Challenge website.
2. The next step is to perform the training and evaluation of models for optic disc segmentation. Using the Disc_Segmentation code, it is possible to train and evaluate different U-NET models from scratch and perform fine-tuning of pre-trained models, in addition to saving the generated masks.
3. Using the Crop_ROI_GAMMA code, it is possible to crop the optical disc region from the masks generated in the previous step. This is a region of interest that has great relevance for grading the stage of glaucoma.
4. The next step consists of building and optimizing one-level models, using GAMMA_OneLevel_Models notebook, which can be 2D, receiving as input the fundus images or just the optical disc region, or 3D receiving as input volumes of OCT images. Each trained model is evaluated using the test set, which results in a csv file containing the classification performed by each model. This file must be submitted for evaluation on the GAMMA challenge website. It is possible to ensemble the best models obtained in the optimization process, to achieve the final result. It is also possible to save the best models, fed with each image modality, and use two ensemble strategies (average and mode) to increase the classification capacity. It is necessary to import the utils_GAMMA_V2 file, which includes functions for reading fundus images and OCT volumes.
5. In a similar way, it is possible to evaluate two levels models, fed with fundus images and with images of the optic disc region. It is possible to train models with two levels, using two feature combination strategies, concatenation and addition.
6. It is also possible to optimize three-level models, fed with fundus images, disc region and OCT volumes. The feature combination strategies mentioned in the previous step can be evaluated.
7. Using the GRAD_CAM code it is possible to interpret the classification made by the models using Grad-CAM (<a href="https://arxiv.org/abs/1610.02391" target="_blank">Grad-cam</a>).
# Results
The Table below presents the results achieved during the model optimization step, taking each of the imaging modalities as input. In some cases, the models were not evaluated due to the limitation of submitting results on the challenge organizers' website. The models were chosen to be evaluated according to the validation loss obtained during the training step. The best results are shown in bold.
![alt text](https://github.com/MarcosMF86/Glaucoma-Grading/blob/main/Results.PNG?raw=true)

The Table below presents the results obtained using ensemble and feature combination.
![alt text](https://github.com/MarcosMF86/Glaucoma-Grading/blob/main/results_ensemble.PNG?raw=true)
# Requirements
* Python 3.9 or later
* Classification models 3D (<a href="https://github.com/ZFTurbo/classification_models_3D" target="_blank">3D Models</a>)
