# library doc string
'''
Author: Ibrahim Sherif
Date: August, 2021
'''

# import libraries
from sklearn.metrics import plot_roc_curve, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib
import shap
import yaml
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from utils import setup_logger
sns.set()
matplotlib.use('Agg')


logger = setup_logger('main_logger', './logs/churn_library.log')

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)


def import_data(file_path):
    """
    Returns dataframe for the csv found at path
    Args:
        file_path (str): The path to the csv file
    Returns:
        df_data (pandas dataframe): Dataframe of the loaded csv file
    """
    df_data = pd.read_csv(file_path, index_col=0)

    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df_data = df_data.drop(['Attrition_Flag'], axis=1)

    return df_data


def perform_eda(df_data):
    """
    Perform eda on df_data and save figures to images folder
    Args:
        df_data (pandas dataframe): df_data for the data to do EDA
    Returns:
        None
    """
    save_path = config['eda']['save_path']

    fig = plt.figure(figsize=(8, 6))
    sns.countplot(x=df_data['Churn'])
    fig.savefig(save_path + config['eda']['plots']['fig_churn_distribution'])

    fig = plt.figure(figsize=(12, 8))
    sns.histplot(x='Customer_Age', hue='Churn', kde=True, data=df_data)
    fig.savefig(save_path + config['eda']['plots']['fig_customer_age_distribution'])

    fig = plt.figure(figsize=(12, 8))
    sns.countplot(x='Marital_Status', hue='Churn', data=df_data)
    fig.savefig(save_path + config['eda']['plots']['fig_marital_status_distribution'])

    fig = plt.figure(figsize=(12, 8))
    sns.countplot(x='Education_Level', hue='Churn', data=df_data)
    fig.savefig(save_path + config['eda']['plots']['fig_education_level_distribution'])

    fig = plt.figure(figsize=(12, 8))
    sns.countplot(x='Income_Category', hue='Churn', data=df_data)
    fig.savefig(save_path + config['eda']['plots']['fig_income_category_distribution'])

    fig = plt.figure(figsize=(12, 8))
    sns.histplot(x='Total_Trans_Ct', hue='Churn', kde=True, data=df_data)
    fig.savefig(save_path + config['eda']['plots']['fig_total_trans_cnt_distribution'])

    fig = plt.figure(figsize=(12, 8))
    sns.histplot(x='Total_Trans_Amt', hue='Churn', kde=True, data=df_data)
    fig.savefig(save_path + config['eda']['plots']['fig_total_trans_amt_distribution'])

    fig = plt.figure(figsize=(12, 8))
    sns.histplot(
        x='Avg_Utilization_Ratio',
        hue='Churn',
        kde=True,
        data=df_data)
    fig.savefig(save_path + config['eda']['plots']['fig_marital_status_x_total_trans_cnt'])

    fig = sns.catplot(
        x='Marital_Status',
        y='Total_Trans_Ct',
        hue='Churn',
        kind='box',
        data=df_data,
        height=5,
        aspect=2)
    fig.savefig(
        save_path +
        config['eda']['plots']['fig_marital_status_x_total_trans_cnt_x_gender'])

    fig = sns.catplot(
        x='Marital_Status',
        y='Total_Trans_Ct',
        hue='Churn',
        row='Gender',
        kind='box',
        data=df_data,
        height=5,
        aspect=2)
    fig.savefig(
        save_path +
        config['eda']['plots']['fig_marital_status_x_total_trans_amt'])

    fig = sns.catplot(
        x='Marital_Status',
        y='Total_Trans_Amt',
        hue='Churn',
        kind='box',
        data=df_data,
        height=5,
        aspect=2)
    fig.savefig(
        save_path +
        config['eda']['plots']['fig_marital_status_x_total_trans_amt_x_gender'])

    fig = sns.catplot(
        x='Marital_Status',
        y='Total_Trans_Amt',
        hue='Churn',
        row='Gender',
        kind='box',
        data=df_data,
        height=5,
        aspect=2)
    fig.savefig(
        save_path +
        config['eda']['plots']['fig_churn_distribution'])

    fig = sns.jointplot(
        x="Total_Trans_Ct",
        y="Total_Trans_Amt",
        hue="Churn",
        data=df_data,
        height=10)
    fig.savefig(
        save_path +
        config['eda']['plots']['fig_total_trans_amt_x_total_trans_cnt'])

    plt.figure(figsize=(16, 6))
    mask = np.triu(np.ones_like(df_data.corr(), dtype=np.bool_))
    heatmap = sns.heatmap(
        df_data.corr(),
        mask=mask,
        vmin=-1,
        vmax=1,
        annot=True,
        cmap='BrBG')
    heatmap.set_title(
        'Triangle Correlation Heatmap',
        fontdict={
            'fontsize': 18},
        pad=16)
    plt.savefig(
        save_path +
        config['eda']['plots']['fig_features_correlation_heatmap'],
        bbox_inches='tight')


def encoder_helper(df_data, category_list):
    """"
    Helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook
    Args:
        df_data (pandas dataframe): df_data for the data to do EDA
        category_list (array like): list of columns that contain categorical features
    Return:
        df_data (pandas dataframe): preprocessed df_data
    """
    for col in category_list:
        mapping_dict = df_data.groupby(col).mean()['Churn']
        df_data[col] = df_data[col].map(mapping_dict)

    return df_data


def perform_feature_engineering(df_data, drop_columns):
    """
    Args:
        df_data (pandas dataframe): df_data for the data to do feature engineering
        drop_columns (list): list of columns to drop
    Returns:
        x_train (pandas dataframe): X training data
        x_test (pandas dataframe): X testing data
        y_train (pandas dataframe): y training data
        y_test (pandas dataframe): y testing data
    """
    # create new feature Total transaction average ticket size
    df_data['Total_Trans_Ats'] = df_data['Total_Trans_Amt'] / \
        df_data['Total_Trans_Ct']

    # get X and y datasets
    x_data = df_data.drop(drop_columns + ['Churn'], axis=1)
    y_data = df_data['Churn']

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=config['data']['test_size'],
        stratify=y_data, random_state=config['random_state'])

    return x_train, x_test, y_train, y_test


def classification_report_image(
        y_train,
        y_test,
        y_train_preds,
        y_test_preds,
        model_name):
    """
    Produces classification report for training and testing results and stores report as image
    in images folder
    Args:
        y_train: training response values
        y_test:  test response values
        y_train_preds: training predictions
        y_test_preds: test predictions
        model_name (str): The name of the model
    Returns:
        None
    """
    plt.figure(figsize=(5, 5))

    plt.text(0.01, 0.6, f"{model_name} Train", {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 1.25, f"{model_name} Test", {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')

    save_path = config['metrics']['save_path']
    plt.savefig(
        f"{save_path}classification_report_{model_name}.png",
        bbox_inches='tight')


def train_and_evaluate_model(model, x_train, x_test, y_train, y_test):
    """
    Train model and save results: images + metrics, and store models
    Args:
        model: Model object
        x_train: X training data
        x_test: X testing data
        y_train: y training data
        y_test: y testing data
    Returns:
        None
    """
    logger.info("Model training")
    model.fit(x_train, y_train)

    logger.info("Model prediction")
    y_train_preds = model.predict(x_train)
    y_test_preds = model.predict(x_test)

    model_name = model.__class__.__name__
    if isinstance(model, Pipeline):
        model_name = model['model'].__class__.__name__

    # calculating ROC scores
    logger.info(
        "%s Train ROC: %s", model_name, roc_auc_score(y_train, y_train_preds))
    logger.info(
        "%s Test ROC: %s", model_name, roc_auc_score(y_test, y_test_preds))

    logger.info("Calculating metrics and saving classification report")
    classification_report_image(
        y_train,
        y_test,
        y_train_preds,
        y_test_preds,
        model_name)

    logger.info("Saving model")
    save_path = config['models']['save_path']
    joblib.dump(model, f"{save_path}{model_name}.pkl")

    return model


def roc_curve_image(x_data, y_data, split_data, *models):
    """
    Plot the roc curve for all models
    Args:
        x_data (pandas dataframe): df_data for the feature data
        y_data (pandas dataframe): df_data for the target variable
        split_data (str): Data split name to compute metrics for
        models (list): List for all models needed to plot the curve
    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    for model in models:
        model_name = model.__class__.__name__
        if isinstance(model, Pipeline):
            model_name = model['model'].__class__.__name__

        # create and save roc curve plots
        plot_roc_curve(
            model,
            x_data,
            y_data,
            ax=ax,
            alpha=0.8,
            name=model_name)

    save_path = config['metrics']['save_path']
    plt.savefig(f"{save_path}{split_data}_roc_auc_curve.png")


def feature_importance_plot(model, x_data):
    """
    Creates and stores the feature importance plot
    Args:
        model (sklearn tree model): model object containing feature_importance
        x_data (pandas dataframe): df_data for the feature data
    Returns:
        None
    """
    save_path = config['metrics']['save_path']
    model_name = model.__class__.__name__
    if isinstance(model, Pipeline):
        model_name = model['model'].__class__.__name__

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_data)

    plt.figure()
    shap.summary_plot(shap_values, x_data, plot_type="bar", show=False)
    plt.title(f"{model_name} Feature Importance")
    plt.savefig(
        f"{save_path}{model_name}_shap_feature_importance.png",
        bbox_inches='tight')

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    fig = plt.figure(figsize=(20, 5))
    plt.title(f"{model_name} Feature Importance")
    plt.ylabel('Importance')

    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    fig.savefig(
        f"{save_path}{model_name}_feature_importance.png",
        bbox_inches='tight')


def main():
    """
    Main function to run script
    """
    logger.info("Loading csv file into dataframe")
    df_data = import_data(config['data']['csv_path'])

    logger.info("Performing and saving EDA plots")
    perform_eda(df_data)

    logger.info("Encoding categorical columns")
    df_data = encoder_helper(df_data, config['data']['categorical_features'])

    drop_columns = ['CLIENTNUM']
    logger.info("Performing feature engineering")
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        df_data, drop_columns)

    logger.info("x_train data shape: %s", x_train.shape)
    logger.info("y_train data shape: %s", y_train.shape)
    logger.info("x_test data shape: %s", x_test.shape)
    logger.info("y_test data shape: %s", y_test.shape)

    logger.info("Training logistic regression model")
    model_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=config['random_state']))
    ])
    model_lr = train_and_evaluate_model(
        model_lr, x_train, x_test, y_train, y_test)

    logger.info("Training random forest model")
    model_rf = RandomForestClassifier(**config['models']['random_forest'])
    model_rf = train_and_evaluate_model(
        model_rf, x_train, x_test, y_train, y_test)

    logger.info("Compute and save roc curve for train data")
    roc_curve_image(x_train, y_train, "Train", model_rf, model_lr)

    logger.info("Compute and save roc curve for test data")
    roc_curve_image(x_test, y_test, "Test", model_rf, model_lr)

    logger.info("Computing feature importances for random forest model")
    feature_importance_plot(model_rf, x_test)


if __name__ == "__main__":
    main()
