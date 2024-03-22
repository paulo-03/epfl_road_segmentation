# Image Processing - Identifying Road Segment From Satellite Images

This repository summarizes our second project undertaken during the Machine Learning course 
([CS-433](https://edu.epfl.ch/coursebook/fr/machine-learning-CS-433)) at the École Polytechnique Fédérale de Lausanne 
([EPFL](https://www.epfl.ch/en/)). The project involves binary categorization of each pixel in an image as either road 
or not road, enabling the segmentation of road sections. The results of our research and the performance achieved in 
this work can be found in the PDF file of this repository.

*Note : Dataset used to train the models can be downloaded [here](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files). Also, 
the four trained models savings to retrieve easily performance at a specific epoch can be found [here](https://drive.google.com/drive/folders/1q1sDpiwFIFM1vTKfZ6gtVwd53aXmacRV?usp=share_link).*

**Authors:** : 
[Luca Carroz](https://people.epfl.ch/luca.carroz), 
[David Schroeter](https://people.epfl.ch/david.schroeter), 
[Paulo Ribeiro de Carvalho](https://people.epfl.ch/paulo.ribeirodecarvalho)

<hr style="clear:both">

## Description

This section is here to guide you through our repository organization and to explain briefly what contains each file or 
why it exists.

- `/ml-project-2-dlp`

    - `/dataset`
        - `/test_set_images`: contains all test images
        - `/training`: contains all train images and respective ground truth
    - `/models`: contains 4 trained model savings to retrieve easily performance at a specific epoch
      - `/d-link_96x96`: savings for DLink Net model on 96x96 images
      - `/log_reg`: savings for Logistic Regression model
      - `/u_net`: savings for UNet model on 400x400 images
      - `/u_net_96x96`: savings for UNet model on 96x96 images
    - `/utils`
      - `cnn.py`: abstract implementation of a CNN
      - `cnn_trainer.py`: used to facilitate training of CNNs
      - `cnn_viewer.py`: used to facilitate displaying information of a CNN
      - `D_Link_Net.py`: implement DLink Net architecture
      - `data_augmentation.py`: used to increase the samples/images in the training data set
      - `feature_extraction.py`: used to extract features from image patches
      - `helpers.py`: implement some useful functions across CNN models
      - `img_transofmration.py`: implement some image transformation (old version of data_augmentation)
      - `lin_reg_crossvalidation.py`: functions to perform cross validation of a logistic regression
      - `mask_to_submission.py`: functions to create submission file for the [AI Crowd contest](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/leaderboards)
      - `performance_metrics.py`: functions to compute all performance metric of our models
      - `random_model.py`: models that make random predictions
      - `U_Net.py`: implement DLink Net architecture

    - `cnn.ipynb`: jupyter notebook summarizing our research journey trough CNN solutions
    - `logistic_reg.ipynb`: jupyter notebook summarizing our research journey trough Logistic Regression solution
    - `model_comparison`: compare the performance metrics of each model
    - `run.py`: script used to ease the training process

## Libraries

- Numpy

`conda install -c conda-forge numpy=1.26.2`\
`pip install numpy==1.26.2`

- Matplotlib

`conda install -c conda-forge matplotlib=3.8.1`\
`pip install -U matplotlib==3.8.1`

- Pytorch

`conda install pytorch::pytorch=2.1.1 torchvision=0.16.1 torchaudio=2.1.1 -c pytorch`\
`pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1`

- Scikit-learn
`conda install -c conda-forge scikit-lean=1.3.2`\
`pip install scikit-lean==1.3.2`

- Seaborn
`conda install -c conda-forge seaborn=0.13.0`\
`pip install seaborn==0.13.0`

## Usage

To generate the submission file for the AI crowd challenge, simply run the python script `run.py`. 
After a while, a file `submission.csv` will be created which contains the predictions.

Feel free to modify the script to change the model, batch size, number of epochs, etc. (see the documentation in the file)
