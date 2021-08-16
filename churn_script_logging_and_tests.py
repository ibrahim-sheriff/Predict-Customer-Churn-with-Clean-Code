'''
Author: Ibrahim Sherif
Date: August, 2021
'''

import logging
import math
import yaml
from sklearn.ensemble import RandomForestClassifier
import churn_library as cls
from utils import setup_logger, check_file_exists


logger = setup_logger('test_logger', './logs/tests_churn_library.log')
logging.getLogger('main_logger').disabled = True

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)


def test_import(file_path):
    """
    Test data import
    """
    try:
        df_data = cls.import_data(file_path)
        logger.info("SUCCESS Reading data")
    except FileNotFoundError as error:
        logger.error("The file wasn't found %s", error)

    try:
        assert df_data.shape[0] > 0
        assert df_data.shape[1] > 0
        logger.info("SUCCESS Data loaded")
    except AssertionError as error:
        logger.error(
            "The file doesn't appear to have rows and columns %s",
            error)

    try:
        assert "Churn" in df_data.columns.values
        logger.info(
            "SUCCESS Churn target variable is found")
    except AssertionError as error:
        logger.error(
            "The dataframe doesn't contain Churn target variable %s",
            error)

    return df_data


def test_eda(df_data):
    """
    Test perform eda function
    """
    try:
        cls.perform_eda(df_data)
        logger.info("SUCCESS eda function worked")
    except KeyError as error:
        logger.error(
            "The dataframe is missing some columns for the eda %s %s",
            error, error)

    save_path = config['eda']['save_path']
    try:
        for _, val in config['eda']['plots'].items():
            assert check_file_exists(save_path, val)
            logger.info("SUCCESS %s plot found", val)
    except AssertionError as error:
        logger.error(
            "%s plot not found in path %s %s",
            val, save_path, error)


def test_encoder_helper(df_data):
    """
    Test encoder helper
    """
    try:
        # Check column presence
        assert set(df_data.columns.values).issuperset(
            set(config['data']['categorical_features']))
        logger.info(
            "SUCCESS All categorical columns are available")
    except AssertionError as error:
        logger.error(
            "Not all categorical variables are available %s",
            error)

    try:
        cls.encoder_helper(df_data, config['data']['categorical_features'])
        logger.info(
            "SUCCESS Categorical variables encoded")
    except KeyError as error:
        logger.info(
            "Encoding categorical variables failed %s",
            error)

    return df_data


def test_perform_feature_engineering(df_data):
    """
    Test perform_feature_engineering function
    """
    drop_columns = ['CLIENTNUM']
    try:
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            df_data, drop_columns)
        logger.info(
            "SUCCESS Feature engineering done")
    except KeyError as error:
        logger.error(
            "There is a problem with feature engineering %s",
            error)

    # Test 'Total_Trans_Ats' feature available or not
    try:
        assert "Total_Trans_Ats" in x_train.columns.values
        assert "Total_Trans_Ats" in x_test.columns.values
        logger.info(
            "SUCCESS Total_Trans_Ats feature found")
    except AssertionError:
        logger.error(
            "Total_Trans_Ats feature not found %s",
            error)

    # Check drop_columns not available
    try:
        assert set(drop_columns).isdisjoint(set(x_train.columns.values))
        assert set(drop_columns).isdisjoint(set(x_test.columns.values))
        logger.info(
            "SUCCESS drop_columns successfully dropped")
    except AssertionError:
        logger.error(
            "drop_columns not dropped")

    # Check target variable 'Churn' correctly placed
    try:
        assert "Churn" not in x_train.columns.values
        assert "Churn" not in x_test.columns.values
        assert "Churn" in y_train.name
        assert "Churn" in y_test.name
        logger.info(
            "SUCCESS Target variable 'Churn' in y data only")
    except AssertionError:
        logger.error(
            "Target variable 'Churn' misplaced")

    # Check y data only has 'Churn' column
    try:
        assert len(y_train.shape) == 1
        assert len(y_test.shape) == 1
        logger.info(
            "SUCCESS y data only contains 'Churn' column")
    except AssertionError:
        logger.error(
            "'Churn' column misplaced")

    # Check correct split % of test size
    try:
        assert math.ceil(
            df_data.shape[0] *
            config['data']['test_size']) == x_test.shape[0]
        assert math.ceil(
            df_data.shape[0] *
            config['data']['test_size']) == y_test.shape[0]
        logger.info(
            "SUCCESS Test data size correct")
    except AssertionError:
        logger.error(
            "Test data size is incorrect")

    return x_train, x_test, y_train, y_test


def test_train_and_evaluate_model(x_train, x_test, y_train, y_test):
    """
    Test train_and_evaluate_model function
    """
    model = RandomForestClassifier(random_state=config['random_state'])
    model_name = model.__class__.__name__
    try:
        model = cls.train_and_evaluate_model(
            model, x_train, x_test, y_train, y_test)
        logger.info("SUCCESS training and evaluating model")
    except BaseException:
        logger.error("Model training/evaluting failed")

    image_name = f'classification_report_{model_name}.png'
    try:
        assert check_file_exists(config['metrics']['save_path'], image_name)
        logger.info("SUCCESS classification report image %s saved", image_name)
    except AssertionError:
        logger.info(
            "classification report image %s not found in path %s",
            image_name,
            config['metrics']['save_path'])

    try:
        assert check_file_exists(
            config['models']['save_path'],
            f'{model_name}.pkl')
        logger.info("SUCCESS model saved %s", f'{model_name}.pkl')
    except AssertionError:
        logger.info(
            "model pickle file %s not found in path %s",
            f'{model_name}.pkl',
            config['models']['save_path'])

    return model


def test_roc_curve_image(x_train, y_train, model):
    """
    Test test_roc_curve_image function
    """
    data_split = "Train"
    try:
        cls.roc_curve_image(x_train, y_train, data_split, model)
        logger.info("SUCCESS computed the roc curve")
    except BaseException:
        logger.error("There was a problem with the roc curve ")

    image_name = f'{data_split}_roc_auc_curve.png'
    try:
        assert check_file_exists(config['metrics']['save_path'], image_name)
        logger.info("SUCCESS roc curve image %s saved")
    except AssertionError:
        logger.info("roc curve image %s not found in path %s",
                    image_name, config['metrics']['save_path'])


def test_feature_importance_plot(model, x_data):
    """
    Test feature_importance_plot function
    """
    model_name = model.__class__.__name__
    try:
        cls.feature_importance_plot(model, x_data)
        logger.info("SUCCESS computed feature importance plot")
    except BaseException:
        logger.error("There was a problem with the feature importance plots")

    image_name = f'{model_name}_shap_feature_importance.png'
    try:
        assert check_file_exists(config['metrics']['save_path'], image_name)
        logger.info(
            "SUCCESS shap feature importance image %s saved",
            image_name)
    except AssertionError:
        logger.info(
            "shap feature importance image %s not found in path %s",
            image_name,
            config['metrics']['save_path'])

    image_name = f'{model_name}_feature_importance.png'
    try:
        assert check_file_exists(config['metrics']['save_path'], image_name)
        logger.info(
            "SUCCESS model feature importance image %s saved",
            image_name)
    except AssertionError:
        logger.info(
            "model feature importance image %s not found in path %s",
            image_name,
            config['metrics']['save_path'])


def run():
    """
    Main function to run script
    """
    logger.info("TESTING import_data")
    df_data = test_import(config['data']['csv_path'])

    logger.info("TESTING perform_eda function")
    test_eda(df_data)

    logger.info("TESTING encoder_helper function")
    df_data = test_encoder_helper(df_data)

    logger.info("TESTING perform_feature_engineering function")
    x_train, x_test, y_train, y_test = test_perform_feature_engineering(
        df_data)

    logger.info("TESTING test_train_and_evaluate_model function")
    model = test_train_and_evaluate_model(x_train, x_test, y_train, y_test)

    logger.info("TESTING roc_curve_image function")
    test_roc_curve_image(x_train, y_train, model)

    logger.info("TESTING feature_importance_plot function")
    test_feature_importance_plot(model, x_test)


if __name__ == "__main__":
    run()
