# DECISION-TREE-IMPLEMENTATION

COMPANY: CODTECH IT SOLUTIONS

NAME: CHUKKA JOSHNA PRIYA

INTERN ID: CT04DN841

DOMAIN: MACHINE LEARNING

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

##
This project is a machine learning desktop application built using Python and the Tkinter library. It allows users to load a dataset (in .csv format), choose a target column, train a Decision Tree Classifier, and visualize the resulting model. The application is designed with a user-friendly Graphical User Interface (GUI) that helps users interact with the system easily without writing code.

The main purpose of this project is to help users understand and apply Decision Tree classification, one of the most popular machine learning algorithms. A Decision Tree is a model that splits the data based on different conditions and helps in making decisions or predictions. It’s widely used for tasks like classification (predicting a label or category) and regression (predicting a number).

Technologies and Libraries Used
Python – The core programming language used.

Tkinter – Python’s built-in library for creating GUI-based applications.

Pandas – Used for reading and processing datasets.

NumPy – Used for numerical operations.

Scikit-learn (sklearn) – Provides the DecisionTreeClassifier, training and testing split, and evaluation functions.

Matplotlib – Used for visualizing the decision tree.

Features of the Application
CSV File Loader
The application provides a button to load a dataset (CSV file). Once the file is loaded, the user can see which file is selected and can choose the target column, which is the output column we want to predict.

Target Selection and Test Size
The user can select any column from the dropdown as the target (dependent) variable. A slider is also provided to select the test size, which decides how much of the dataset will be used for testing. For example, a test size of 0.3 means 30% of the data will be used for testing and 70% for training.

Training the Model
On clicking the Train and Visualize button, the app splits the dataset, trains the Decision Tree classifier, and shows the accuracy and a classification report which includes precision, recall, F1-score, and support. These are standard evaluation metrics in machine learning.

Tree Visualization
After training, a tree diagram is shown using Matplotlib, helping the user understand how the model is making decisions.

Export Tree Image
The user can save the decision tree as a PNG image using the “Export Tree” button.

Predict New Record
There’s also a feature to predict the result of a new record. A new window opens where the user can enter values for all features (input columns). Once submitted, the model predicts and displays the result using the trained decision tree.

Advantages and Use Cases
The application is easy to use even for people without strong programming knowledge.

It helps students and professionals understand how decision trees work in practice.

Can be used for educational purposes, demonstrations, and quick experiments with datasets.

Can classify real-world datasets like customer data, health data, or student performance.

Conclusion
This Decision Tree Classifier GUI application is a simple and effective tool for performing classification tasks using a visual interface. It combines Python's data science libraries with a GUI to make machine learning more accessible. With features like model training, evaluation, visualization, and prediction, it provides a full workflow for beginners to explore machine learning in an interactive way.

##

#OUTPUT

![Image](https://github.com/user-attachments/assets/44b850e5-911c-4854-b10a-659b0c7f00e1)
![Image](https://github.com/user-attachments/assets/a431c547-f70a-4247-a321-6618dc0d9d0c)
