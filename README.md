# Decision Trees for Classification on Adult Census Data

This project applies Decision Tree classifiers to predict categorical demographic attributes from the Adult Census dataset. The task involves using features such as age, education.num, marital.status etc to predict target variables like income and relationship status.

## Project Setup

### Step 1: Clone the repository

Clone the repository by running the following command in your terminal or Git Bash:

```
cd your_folder  
git clone "https://github.com/Awesome-AI-ML-DELTA25/TabularDataML.git"
```


### Step 2: Install Conda (if not already installed)

Ensure you have installed either anaconda or miniconda.
[Conda Download](https://www.anaconda.com/download/)


### Step 3: Create and activate the Conda environment

To set up the same virtual environment as used in this project, run the following command:  
` conda env create -f environment.yml `

Activate the environment:  
` conda activate weekII_env `


### Step 4: Install required packages
 ` pip install -r /path/to/requirements.txt ` 

## Overview

Decision trees split data based on feature thresholds. They are widely used for their interpretability and ability to handle both numerical and categorical data. In this project, we use decision trees to predict:

1. Income
2. Marital.status
3. Relationship
4. Workclass
5. Sex
6. Race

Points 5 and 6 cannot be predicted in an ideal world, but due to biases present in society, we try to see what the decision tree uncovers .

The dataset is preprocessed using One-Hot Encoding for categorical columns, and models are trained using scikit-learn pipelines for cleaner workflows.

## Preprocessing

| Column         | Type    | Encoding         |
| -------------- | ------- | ---------------- |
| workclass      | Nominal | One-Hot Encoding |
| marital.status | Nominal | One-Hot Encoding |
| occupation     | Nominal | One-Hot Encoding |
| relationship   | Nominal | One-Hot Encoding |
| race           | Nominal | One-Hot Encoding |
| sex            | Binary  | One-Hot Encoding |
| native.country | Nominal | One-Hot Encoding |
| income         | Binary  | One-Hot Encoding |

Dummy variables are generated and concatenated with the original dataset for model training.


## Model and Predictions

### Income

- Model: DecisionTreeClassifier (max_depth=10, min_samples_split=50)
- Evaluation:
    - Accuracy: 0.8589743589743589
    - Precision: 0.7959372114496768
    - Recall: 0.5586519766688269
    - F1: 0.6565118050266565

#### Confusion Matrix:

![Confusion Matrix of Income](images\Income_Confusion_Matrix.png)

1. Accuracy is the best measurement for income, since any wrong classification of income is equally bad.
2. The true negatives are highest in number.
3. The recall of >50K is bad (0.56).


#### Feature Importance:

![Feature Importance of Income](images\Income_Features.png)

It only has the top 10 most contributing features to prevent cluttering.

#### Decision Tree:

![Decision Tree of Income](images\Income_Decision_Tree.png)

Marital.status being married-civ-spouse has a large impace on the income, probably due to factors like tax benefits and higher investing power through combined wealth. This is also reflected in the decision tree.

### Relationship

- Model: DecisionTreeClassifier (max_depth=9, criterion='entropy', min_samples_split=900)
- Evaluation:
    - Accuracy: 0.7851782363977486
    - Precision: 0.6208051836626334
    - Recall   : 0.6434882567321479
    - F1 Score : 0.6312090846356724

#### Confusion Matrix:

![Confusion Matrix of Relationship](images\Relationship_Confusion_Matrix.png)

1. Accuracy is the best measurement for income, since any wrong classification of relationship is equally bad.
2. The recall for 'wife' is worst at 0.49

#### Decision Tree:

![Decision Tree of Relationship](images\Relationship_Decision_Tree.png)

Marital.status being married-civ-spouse has a large impace on the income, probably due to factors like tax benefits and higher investing power through combined wealth. This is also reflected in the decision tree.


You can view the graphs for other features under images or in the jupter file (tabular_data.ipynb).

Since the dependency of marital.status on relationship was quite high and there can be some easy giveaways like in case of husband and wife, we also trained a model after dropping marital.status.

### Relationship (Dropping Marital Status)

## Author
Aditey Nandan  