# MNIST-handwritten-images
**Introduction**

This project explores the classification of handwritten digits (0-9) using the MNIST dataset. Various machine learning and deep learning models were implemented and evaluated, with the Convolutional Neural Network (CNN) demonstrating the best performance.


---

**Features**

- Data preprocessing: Flattening, reshaping, and normalization.

- Models applied: SVM, KNN, ANN, and CNN.

- Hyperparameter tuning for SVM and KNN.

- Achieved a maximum accuracy of 98.81% using CNN.



---

Dataset

The MNIST dataset consists of 70,000 grayscale images of handwritten digits:

Training Set: 48,000 images

Validation Set: 12,000 images

Image size: 28x28 pixels


Key Characteristics:

Clean dataset with no missing values or outliers.

Scaling handled through normalization.



---

Project Workflow

1. Data Preprocessing:

Normalized data for SVM and KNN (flattened into 1D vectors).

Reshaped data for CNN to maintain 2D spatial structure (28x28x1).



2. Model Implementation:

Support Vector Machine (SVM): Achieved 97.7% accuracy with hyperparameter tuning.

K-Nearest Neighbors (KNN): Achieved 96.81% accuracy with hyperparameter tuning.

Artificial Neural Networks (ANN): Achieved 92.45% accuracy.

Convolutional Neural Networks (CNN): Achieved 98.81% accuracy.



3. Model Evaluation:

Metrics: Accuracy, confusion matrix, and performance comparison.



4. Conclusion:

CNN emerged as the best model, demonstrating its superiority in capturing spatial patterns and hierarchies in image data.

While SVM performed well, it was less robust than CNN for image classification tasks.





---

Key Insights

1. CNN Dominance:

CNN's ability to capture local patterns and spatial hierarchies makes it the most suitable model for image-related tasks like MNIST.



2. SVM Performance:

SVM performed well but lacked the robustness and accuracy of CNN for this dataset.



3. Simpler Models (KNN, ANN):

These models showed decent accuracy but are less effective than CNN for image classification tasks.





---

Results

SVM Accuracy: 97.7%

KNN Accuracy: 96.81%

ANN Accuracy: 92.45%

CNN Accuracy: 98.81%



---

Requirements

Install the following dependencies to run the project:

pip install numpy pandas matplotlib seaborn scikit-learn tensorflow


---

Usage

1. Clone the repository:

git clone https://github.com/yourusername/mnist-handwritten-images.git  
cd mnist-handwritten-images


2. Run the code:

python main.py




---

Contributing

Feel free to open issues or submit pull requests for suggestions or improvements.


---

License

This project is licensed under the MIT License.


---

Acknowledgements

MNIST dataset: Yann LeCun's Website

Tutorials and resources that guided the project.
