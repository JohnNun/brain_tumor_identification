# Brain Tumor Identification and Classification
![](/images/pexels-cottonbro-5723883[crop].jpg)
The focus of this project is the creation and development of a machine learning algorithm that can properly identify and categorize brain tumors through MRI imagery, possibly leading to faster diagnosis. This project uses the Brain Tumor Dataset, sourced and compiled from various smaller datasets from on Kaggle and Figshare, comprised by approximately 10,500 MRI images categorized into four classes: 'no_tumor', 'glioma', 'pituitary', and 'meningioma'. 

For the development of the model, tools and modules from TensorFlow’s Keras library were utilized. Models used for testing included DenseNet and ResNet models, alongside custom Convolutional Neural Network (CNNs) model. After comparing and evaluation results, the DenseNet model was chosen for further development and tuning due to its overall performance with fewer layers. As the data is imbalanced this can lead to less than favorable results, so to mitigate overfitting, enhance generalization, and give the minority class a proper chance, several regularization methods were implemented such as: dropout layers randomly drop 50% of nodes during training, and L2 kernel regularizer that penalized the majority class. Additionally, an early stop callback method was implemented to monitor validation loss, stopping training if there were no improvements over three consecutive epochs.The learning rate used for these and the final model is the Adam optimizer as its adaptive learning rate capabilities, and combining momentum and RMSprop techniques accelerate convergence. 

Through these methods, the model aims to effectively learn from the Brain Tumor Dataset, improving identification and classification for the tumor classes present in the dataset leading to faster diagnosis and possibly prompt treatment.

# Business Understanding
According the [Clevland Clinic](https://my.clevelandclinic.org/health/diseases/6149-brain-cancer-brain-tumor), a brain tumor is an abnormal growth or mass of cells in or around your brain. Brain tumors can be malignant (cancerous) or benign (noncancerous). Some tumors grow quickly, while others grow slowly with only about one-third of brain tumors being cancerous. 

When it comes to manually analyzing and identifying a tumor in the brain, there is always the possibility of misinterpretation due to the medical professional’s fatigue or even lack of time. Not to mention manual analysis of medical images is a slow and labor-intensive process, which can possibly lead to delays in diagnosis. With the development and proper training of an image classification neural network learning algorithm, not only may it be possible to detect and classify brain tumors, but a machine learning model can analyze the medical image in a timely manner leading to faster diagnosis and possibly leading to prompt treatment.

# Data Understanding
The dataset used for this project is the [Brain Tumor Dataset](https://www.kaggle.com/datasets/ishans24/brain-tumor-dataset) from Kaggle datasets, this dataset contains about 10,500 images of Brain MRI scans compiled by Kaggle user [Ishan Singh](https://www.kaggle.com/ishans24) from multiple publicly available datasets on [Kaggle](https://www.kaggle.com) and [Figshare](https://figshare.com/). The data is divided between 4 classes, 'no_tumor', 'glioma', 'pituitary', and 'meningioma'. The rough data distribution is as follows, glioma 3700 images, meningioma 2300 images, no tumor 1700 images, and pituitary at 2700 images, so we can see that the data is imbalanced, primarily with the ‘no_tumor’ class. While having balanced data would be ideal, with the proper regularizations and node dropout the model should be able to overcome this imbalance or at the very least properly learn from the minority class.

## Data preparation
When downloading the data, it comes divided into 4 different directories based off the data’s classes. To make it easier to manage the data I created a parent directory to store all the data. To split the data for model training, I used the splitfolders tool to randomly split the parent directory folder I previously created into train, test, and validation directories with a ratio of .70 for the training data and .15 for both the test and validation data. With the train, test, and validation directories now created I import the data using ImageDataGenerator with a rescale parameter to scale the data as its being imported. By scaling the data as its being imported it allows us to bring in larger amounts of data in and makes it easier to process when creating the train, test, and validation datasets.

# Modeling
For the model building, I imported tools from the Tensorflow library, primarily Keras tools. 
From Tensorflows’s Keras I imported the models I would test and use such as DenseNet, and ResNet as well as the layer modules I’d use to help tune the model such as dense, and dropout layers to name a few. 

Before choosing a final model I created 3 base models, a Convolutional Neural Network (CNN), DenseNet, and ResNet model. From these models I went with the DenseNet model to further fine tune as it showed the best results with the least number of layers. To help with the model tuning, I implemented an early stop callback to keep track of the model’s validation loss and stop the model should its validation loss no longer decrease or start to consecutively increase over a span of 3 epochs. 

The learning rate I chose for this model is Adaptive Moment Estimation (Adam). I went with this optimizer as it uses momentum as well as RMSprop to automatically adjust the models learning rate leading to faster convergence. As the data used for this model is imbalanced the model would begin to overfit, I implemented dropout layers to randomly drop 50% of its layers. I also added L2 kernel regularizers to penalize the majority class in the data so the model can better learn and understand the data from the minority classes lowering overfitting as much as possible.

# Evaluation
The final model performed very well across classification metrics, for precision the model averaged a .98, recall .99, and f1score .98. When evaluating the test data with the final model it was able to properly identify and categorize the data with 98% accuracy and about .15 loss when making its predictions. But where the model really excels is in its ability to recall what it learned, which is the most important metric when it comes to image identification and classification.

For the tumor positive classes ('glioma', 'meningioma’, 'pituitary'), the recall rates were all close to each other, with the rates of the three classes averaging at a .98 recall rate. Further individual breakdown for each tumor positive class recall rate, glioma .98, meningioma .97, and pituitary .1.0. When it comes to the ‘no tumor’ class the model has recall rate of 1.0 meaning that despite being the minority class the model is able to accurately identify and classify brain scans without tumors 100% of the time. 

Overall, the model performed exceptionally well with all of the classification metrics ratings ranging from the mid to upper 90s. This tells us that the model is precise with its predictions and can recall what it learned well even with an imbalanced dataset.

## Limitations
A limitation when working with this data would be Image inconsistency. By inconsistency I mean that for the **majority of the ‘no tumor’ class images are in greyscale** while the **other 3 classes have blue hues and tints**. An inconsistency like this can easily be picked up by the model possibly leading to misclassification of the few images in greyscale for the majority classes.

A second limitation I encountered was the data’s class imbalance. While the class weight imbalance was minimized through dropout layers and kernel regularization this **can lead to slower run times** as well as the model encountering **difficulties generalizing** as the model was forced to ignore data for simply being the majority.

## Next Steps
While the model did perform exceptionally well there may still be **room for improvement** in certain aspects of the model, primarily in its **ability to generalize consistently** as during training the model’s results would have minor fluctuations with the occasional big spike, but this step can be ignored if I am able to procure a more balanced dataset.

As the model was trained with MRIs It is possible that this model could work well with CT (Cat) scans as CT scans are considered the step below MRIs as they are quick scans and provide less detail. To my understanding, patients will typically get a CT scan before an MRI scan, so if the data is available, **running and testing this model with CT scans** can help get an understanding of the models capabilities and if the model performs well with CT scans it may be possible to not only **cut down on the time it takes to get a diagnosis** but it may also be more **cost efficient as MRI scans are typically more expensive monetary and time wise**, and typically demand more from the patient to obtain.

# Conclusion
Overall, the final model shows exceptional performance across all classification metrics, achieving high precision, recall, and F1 scores. The model is able to accurately identify both tumor-positive and tumor-negative cases, even with an imbalanced dataset, highlighting its capability and reliability for image classification tasks.

## Repository Structure
* images
* README.md
* [Notebook](Notebook.ipynb)
* [Presentation](Presentation.pdf)