# IMAGE CLASSIFICATION MODEL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: JYOTSANA BHARDWAJ

*INTERN ID*: CT08DK599

*DOMAIN*: MACHINE LEARNING

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH

🧠 CIFAR-10 Image Classification using CNN
📘 Introduction
This project focuses on building a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The dataset includes 60,000 color images, each measuring 32x32 pixels, in 10 different classes, with 6,000 images per class. CNNs are a strong deep learning method often used for image classification tasks.

I worked on this project as part of my internship at CodTech as Machine Learning intern to gain practical experience with TensorFlow, model building, training, and performance evaluation using deep learning.

🎯 Objective
The goal of this project is to:

- Build a CNN model using TensorFlow/Keras
- Train the model on the CIFAR-10 dataset
- Visualize the training performance, including accuracy and loss
- Evaluate the model using a classification report and confusion matrix
- Understand and interpret the results of deep learning-based image classification

🧰 Libraries and Tools Used
- Python 3  
- TensorFlow/Keras for model creation and training  
- NumPy for data manipulation  
- Matplotlib for plotting graphs  
- Seaborn for heatmap visualizations  
- Scikit-learn for evaluation metrics like the classification report and confusion matrix  

🗂 Dataset: CIFAR-10
The CIFAR-10 dataset serves as a standard benchmark for image classification. It includes the following 10 classes:  

- Airplane  
- Automobile  
- Bird  
- Cat  
- Deer  
- Dog  
- Frog  
- Horse  
- Ship  
- Truck  

Each image is in RGB format and measures 32x32 pixels.

🧪 Model Architecture
- 3 Convolutional Layers with ReLU activation  
- Max Pooling after the first two convolutional layers  
- Flatten Layer to convert the 3D output into 1D  
- Dense Layer with 64 neurons and ReLU  
- Output Layer with 10 neurons, using Softmax for multi-class classification  

The model compiles with the Adam optimizer and uses sparse categorical crossentropy for the loss function.

📈 Evaluation and Results
The model trained for 10 epochs using both training and validation data. Final test accuracy prints after evaluation. The classification report provides precision, recall, and F1-score. A confusion matrix visualizes using Seaborn’s heatmap. These metrics help show how well the model performs across all 10 classes.

🖥️ How to Run
Make sure you have Python and pip installed. Run the script:
<pre><code>python cifar10_cnn.py</code></pre>
