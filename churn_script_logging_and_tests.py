import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def test_import(file_path):
    """
    Test data import
    """
    try:
        df_data = cls.import_data(file_path)
        logging.info("Testing import_data: SUCCESS Reading data")
    except FileNotFoundError as error:
        logging.error("Testing import_eda: The file wasn't found %s", error)
  
    try:
        assert df_data.shape[0] > 0
        assert df_data.shape[1] > 0
        logging.info("Testing import_data: SUCCESS Data loaded")
    except AssertionError as error:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns %s", error)
 
    try:
        assert "Churn" in df_data.columns.values
        logging.info("Testing import_data: SUCCESS Churn target variable is found")
    except AssertionError as error:
        logging.error("Testing import_data: The dataframe doesn't contain Churn target variable %s", error)


def test_eda(perform_eda):
    """
    Test perform eda function
    """
    df_data = cls.import_data("./data/bank_data.csv")
    try:
        perform_eda(df_data)
        logging.info("Testing perform_eda: SUCCESS")
    except KeyError as error:
        logging.error("Testing perform_eda: The dataframe is missing some columns for the eda %s", error)
    
    try:
        assert os.path.exists(os.path.join(os.path.abspath('./images/EDA'), 'churn_distribution.png'))
        logging.info("Testing perform_eda: churn_distribution.png file found")
        assert os.path.exists(os.path.join(os.path.abspath('./images/EDA'), 'customer_age_distribution.png'))
        assert os.path.exists(os.path.join(os.path.abspath('./images/EDA'), 'marital_status_distribution.png'))
        assert os.path.exists(os.path.join(os.path.abspath('./images/EDA'), 'education_level_distribution.png'))
        assert os.path.exists(os.path.join(os.path.abspath('./images/EDA'), 'income_category_distribution.png'))
        assert os.path.exists(os.path.join(os.path.abspath('./images/EDA'), 'total_transaction_count_distribution.png'))
        assert os.path.exists(os.path.join(os.path.abspath('./images/EDA'), 'total_transaction_amount_distribution.png'))
        assert os.path.exists(os.path.join(os.path.abspath('./images/EDA'), 'marital_status_x_total_transaction_count_box_plot.png'))
        assert os.path.exists(os.path.join(os.path.abspath('./images/EDA'), 'marital_status_x_total_transaction_count_x_gender_box_plot.png'))
        assert os.path.exists(os.path.join(os.path.abspath('./images/EDA'), 'marital_status_x_total_transaction_amount_box_plot.png'))
        assert os.path.exists(os.path.join(os.path.abspath('./images/EDA'), 'marital_status_x_total_transaction_amount_x_gender_box_plot.png'))
        assert os.path.exists(os.path.join(os.path.abspath('./images/EDA'), 'total_transaction_amt_x_total_transaction_cnt_scatter_plot.png'))
        assert os.path.exists(os.path.join(os.path.abspath('./images/EDA'), 'features_correlation_heatmap.png'))
        
    except:
        pass


def test_encoder_helper(encoder_helper):
    """
    Test encoder helper
    """
    df_data = cls.import_data("./data/bank_data.csv")
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    try:
        # Check column presence
        assert set(df_data.columns.values).issuperset(set(cat_columns))
        logging.info("Testing encoder_helper: SUCCESS All categorical columns are available")
    except AssertionError as error:
        logging.error("Testing encoder_helper: Not all categorical variables are available %s", error)
 
    try:
        encoder_helper(df_data, cat_columns)
        logging.info("Testing encoder_helper: SUCCESS")
    except KeyError as error:
        logging.info("Testing encoder_helper: The dataframe is missing a categorical column %s", error)
 

def test_perform_feature_engineering(perform_feature_engineering):
    """
    Test perform_feature_engineering
    """
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    numeric_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]

    df_data = cls.import_data("./data/bank_data.csv")
    df_data = cls.encoder_helper(df_data)
    try:
        perform_feature_engineering(df_data, cat_columns, numeric_columns)
        logging.info("Testing perform_feature_engineering: SUCCESS Feature engineering done")
    except KeyError as error:
        logging.error("Testing perform_feature_engineering: There is a problem with feature engineering %s", error)


def test_train_models(train_models):
    """
    Test train_models
    """



def run():
    test_import("./data/bank_data.csv")
    test_eda()
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_eda)

if __name__ == "__main__":
    run()








