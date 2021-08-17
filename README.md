# Predict Customer Churn with Clean Code

The first project for the [ML DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) by Udacity.

## Description

This project is part of Unit 2: Clean Code Principles. The problem is to predict credit card customers that are most likely to churn using clean code best practices.

## Prerequisites

Python and Jupyter Notebook are required.
Also a Linux environment may be needed within windows through WSL.

## Dependencies
- sklearn
- numpy
- pandas
- matplotlib
- seaborn
- shap

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies from the ```requirements.txt```

```bash
pip install -r requirements.txt
```

## Usage

The entire process can be checked in the jupyter notebook **churn_notebook.ipynb**

The main script to run using the following command.
```bash
python churn_library.py
``` 
which will generate
- EDA plots in the directory ```./images/EDA/```
- Model metrics plots in the directory ```./images/metrics/```
- Saved model pickle files in the directory ```./models/```
- A log file ```./log/churn_library.log``` 

The tests script can be used with the following command which will generate a log file ```./log/tests_churn_library.log``` 
```bash
python churn_script_logging_and_tests.py
```

## License
Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See ```LICENSE``` for more information.