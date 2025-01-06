Classification of Conditions Using Gene Expression

This project involves analyzing gene expression data to classify biological conditions using machine learning models. The project utilizes statistical methods for feature selection, and Random Forest classifiers for prediction, with hyperparameter tuning for optimization.
Project Overview
Objectives:

    Preprocess gene expression data to normalize and prepare for analysis.
    Select significant features using statistical tests like ANOVA.
    Train a Random Forest classifier for condition classification.
    Optimize the model using hyperparameter tuning with GridSearchCV.
    Evaluate the model using metrics like accuracy, precision, recall, and F1-score.

Dataset

The dataset used in this project contains gene expression values (FPKM) for different conditions:

    CTR: Control samples.
    DSF: Disease factor samples.
    IM: Immunotherapy samples.
    IM.DSF: Immunotherapy with disease factor samples.

Source: The dataset is uploaded as GSE276122_FPKMtable.csv.
Steps in the Workflow
1. Data Preprocessing

    Standardized expression data using StandardScaler.
    Extracted features (CTR_FPKM, DSF_FPKM, IM_FPKM, IM.DSF_FPKM) and labels (condition).
    Split data into training and testing sets.

2. Feature Selection

    Performed ANOVA to identify genes with significant differences across conditions.
    Retained features with p-value < 0.05.

3. Model Training

    Trained a Random Forest Classifier on the training data.

4. Hyperparameter Tuning

    Optimized hyperparameters (e.g., n_estimators, max_depth) using GridSearchCV.
    Selected the best model for improved performance.

5. Model Evaluation

    Evaluated the model using metrics:
        Accuracy
        Precision
        Recall
        F1-score
    Visualized results using confusion matrices and feature importance plots.

Technologies Used

    Python Libraries:
        pandas and numpy: Data manipulation and analysis.
        scikit-learn: Machine learning and evaluation.
        seaborn and matplotlib: Data visualization.

How to Run the Project

    Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Install required libraries:

pip install -r requirements.txt

Open and run the Jupyter Notebook or Python script:

    Google Colab: Upload the .ipynb file to your Colab workspace.
    Local Environment: Run the script using:

        python main.py

    View the results and visualizations.

Results
Model Performance

    Baseline Accuracy: 85%
    Optimized Model Accuracy: 97%

Visualizations

    Confusion Matrix
    Feature Importance Plot
    Gene Expression Distributions

Project Files
2114951040_BioInformatics.ipynb
Classification of Conditions Using Gene Expression_2114951040.docx
Classification of Conditions Using Gene Expression_2114951040.pdf
GSE276122_FPKMtable.csv


Contributors

    ISHRAQ UDDIN CHOWDHURY
    Bioinformatics Student
    [Linkedin](https://www.linkedin.com/in/ishraqinventor/) | [GitHub Profile](https://github.com/ishraqX/)

License

This project is licensed under the MIT License. See the LICENSE file for details.
