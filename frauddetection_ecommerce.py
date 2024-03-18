import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils_fraud import DataLoader, DataPreprocessor, DataAnalysis, DataVisualizer, FeatureEngineering, ModelUtils, \
    Model, FineTuningModel

filepath='./data/Customer_DF.csv'

dataloader= DataLoader(filepath)

df_customer=dataloader.load_data()

print(df_customer.columns)
print(df_customer.info())
print(df_customer.describe())

filepath1 ='./data/cust_transaction_details.csv'

dataloader1 = DataLoader(filepath1)

df_transactions=dataloader1.load_data()

print(df_transactions.columns)
print(df_transactions.info())
print(df_transactions.describe())

# Preprocess the datasets
customer_preprocessor = DataPreprocessor(df_customer)
transaction_preprocessor = DataPreprocessor(df_transactions)

column_name = {'Unnamed: 0' : 'SN'}
df_customer=customer_preprocessor.rename_columns(column_name)
df_transactions=transaction_preprocessor.rename_columns(column_name)

customer_preprocessor.set_index('SN')
transaction_preprocessor.set_index('SN')

customer_preprocessor.check_missing_values()
transaction_preprocessor.check_missing_values()

print(df_customer.head())
print(df_transactions.head())

df = df_customer.copy()
df1= df_transactions.copy()

customer_dataanalysis = DataAnalysis(df)
transaction_dataanalysis = DataAnalysis(df1)

unique_count_cus = customer_dataanalysis.count_unique_col('customerEmail')
unique_count_tra = transaction_dataanalysis.count_unique_col('customerEmail')
print(unique_count_cus)
print(unique_count_tra)

email_counts=customer_dataanalysis.count_items('customerEmail')
print(email_counts)

unique_payment_methods = transaction_dataanalysis.get_unique('paymentMethodType')
print(unique_payment_methods)

uncommon_emails = customer_dataanalysis.find_uncommon_emails(df, df1)
print(f"Number of unique uncommon emails: {len(uncommon_emails)}")
print(uncommon_emails)

common_emails = customer_dataanalysis.find_common_emails(df, df1)
print(f"Number of unique common emails: {len(common_emails)}")


# Visualize data
visualizer = DataVisualizer()
        

visualizer.plot_count(df1, 'paymentMethodType', 'Payment method type')

visualizer.plot_count(df1, 'orderState', 'Order state')

visualizer.plot_count(df1, 'paymentMethodProvider', 'Payment method provider', figsize=(13, 6))

visualizer.plot_count(df1, 'paymentMethodProvider', 'paymentMethodProvider with RegistrationFailure', 
                      hue='paymentMethodRegistrationFailure', figsize=(15, 5))

visualizer.plot_count(df, 'No_Payments', 'No_Payments with Fraud', 
                      hue='Fraud', figsize=(15, 5))

visualizer.plot_boxplot(df_customer, 'No_Orders', 'Num of Orders')

df_final = df[df['customerEmail'].isin(df1['customerEmail'])]

df_final.reset_index(inplace = True)

print(df_final.info())
print(df_final.head())


feature_eng=FeatureEngineering(df_final, df1, 'customerEmail', 'customerEmail')

df_final['Total_Transaction_Amt'] = feature_eng.get_aggregate_values('transactionAmount')

df_final['Avg_Transaction_Amt'] = feature_eng.get_aggregate_values('transactionAmount', aggregate='avg')

df_final['Max_Transaction_Amt'] = feature_eng.get_aggregate_values('transactionAmount', aggregate='max')

df_final['No_Transactions_Failed'] = feature_eng.get_aggregate_values('transactionFailed')

df_final['Payment_Regist_Fail']  = feature_eng.get_aggregate_values('paymentMethodRegistrationFailure')

df_final['Duplicate_IP']  =  feature_eng.count_duplicates(df_final, 'customerIPAddress').tolist()

df_final['Duplicate_Address']  =  feature_eng.count_duplicates(df_final, 'customerBillingAddress').tolist()

df_final['Fraud_labels']  = df_final['Fraud'].astype(int).tolist()

df_final['Paypal_Payments']  = feature_eng.add_feature_column('paymentMethodType', 'paypal')
df_final['Apple_Payments']  = feature_eng.add_feature_column('paymentMethodType', 'apple pay')
df_final['Bitcoin_Payments']  = feature_eng.add_feature_column('paymentMethodType', 'bitcoin')
df_final['Card_Payments']  = feature_eng.add_feature_column('paymentMethodType', 'card')

df_final['Mastercard']  = feature_eng.add_feature_column('paymentMethodProvider','Mastercard')
df_final['VISA_16']  = feature_eng.add_feature_column('paymentMethodProvider','VISA 16 digit')
df_final['VISA_13']  = feature_eng.add_feature_column('paymentMethodProvider','VISA 13 digit')
df_final['AmericanExp']  = feature_eng.add_feature_column('paymentMethodProvider','American Express')
df_final['Discover']  = feature_eng.add_feature_column('paymentMethodProvider','Discover')
df_final['JCB_16']  = feature_eng.add_feature_column('paymentMethodProvider','JCB 16 digit')
df_final['JCB_15']  = feature_eng.add_feature_column('paymentMethodProvider','JCB 15 digit')
df_final['DC_CB']  = feature_eng.add_feature_column('paymentMethodProvider','Diners Club / Carte Blanche')
df_final['Voyager']  = feature_eng.add_feature_column('paymentMethodProvider','Voyager')
df_final['Maestro']  = feature_eng.add_feature_column('paymentMethodProvider','Maestro')

df_final['Orders_Fulfilled'] = feature_eng.add_feature_column('orderState','fulfilled')
df_final['Orders_Pending'] = feature_eng.add_feature_column('orderState','pending')
df_final['Orders_Failed'] = feature_eng.add_feature_column('orderState','failed')

# Condition for transactions failed and order fulfilled
def condition(df):
    return df[(df['orderState'] == 'fulfilled') & (df['transactionFailed'] == 1)]

df_final['Trans_failed_order_fulfilled']  = feature_eng.get_aggregate_values(condition=condition)

print(df_final.info())
print(df_final.head())

print(df_final['Fraud'].sum())


final_dataanalysis = DataAnalysis(df_final)

unique_count_phone = final_dataanalysis.count_unique_col('customerPhone')
print(unique_count_phone)
unique_count_device = final_dataanalysis.count_unique_col('customerDevice')
print(unique_count_device)
unique_count_ip = final_dataanalysis.count_unique_col('customerIPAddress')
print(unique_count_ip)
unique_count_address = final_dataanalysis.count_unique_col('customerBillingAddress')
print(unique_count_address)

dup_ip=final_dataanalysis.get_multi_occurance('Duplicate_IP')
print(dup_ip)

dup_add=final_dataanalysis.get_multi_occurance('Duplicate_Address')
print(dup_add)

visualizer.plot_barplot_trn(data=df_final, x='No_Transactions', y='No_Transactions_Failed', 
                            title='No_Transactions No_Transactions_Failed by Fraud', hue='Fraud')

visualizer.plot_count(df_final, 'Orders_Fulfilled', 'Orders Fulfilled with Fraud', hue ='Fraud')

visualizer.plot_scatterplot(df_final, 'Total_Transaction_Amt', 'No_Transactions_Failed',
                            'No_transactionsFail vs. Total_transaction_amt by Fraud_labels', hue='Fraud_labels', figsize=(12, 5))

visualizer.plot_boxplot_target(df_final, 'Fraud_labels', 'No_Transactions', 
                               'No_Transactions Distribution by Fraud_labels', figsize=(12, 5))



final_preprocess=DataPreprocessor(df_final)

column_name=['SN','customerEmail','customerPhone', 'customerDevice', 'customerIPAddress', 'customerBillingAddress', 'Fraud', 'Fraud_labels']
features = final_preprocess.drop_columns(column_name)
target = df_final['Fraud_labels']

print(features.info())

print(features.shape)
print(target.shape)

# Descriptive Statistics
print("Descriptive Statistics:")
print(features.describe())
print("\n")

visualizer.plot_distribution_individual_features(features)

# ================================================================


mutils=ModelUtils(features, target)

X_train, X_test, y_train, y_test=mutils.split_data()


print(X_train.head())
print(y_train.head())


model_name_svc = 'svc'
param_svc = {
    'random_state': 42
    }

model_name_dt = 'decisiontree'
param_dt = {
    'random_state':0
    }

model_name_xgb='xgboost'
param_xgb={
    'n_estimators': 100, 
    'random_state': 42
    }

model_name_rf='randomforest'
param_rf = {
    'n_estimators': 300,
    'random_state': 42
    }


model_svc=Model(model_name_svc, param_svc)

model_svc.train(X_train, y_train)

y_pred_svc=model_svc.predict(X_test)

score_svc, cmt_svc = model_svc.evaluate(y_test, y_pred_svc)

print(f'Accuracy Score for SVC:{score_svc}')

print(f"""Confusion Matrix SVC:
    {cmt_svc}""")

visualizer.plot_conf_matrix(y_test, y_pred_svc, 'Confusion Matrix in percent for SVC', figsize=(10, 8))

model_dt=Model(model_name_dt, param_dt)

model_dt.train(X_train, y_train)

y_pred_dt=model_dt.predict(X_test)

score_dt, cmt_dt = model_dt.evaluate(y_test, y_pred_dt)

print(f'Accuracy Score for DT:{score_dt}')

print(f"""Confusion Matrix DT:
    {cmt_dt}""")

visualizer.plot_conf_matrix(y_test, y_pred_dt, 'Confusion Matrix in percent for DT', figsize=(10, 8))

model_rf=Model(model_name_rf, param_rf)

model_rf.train(X_train, y_train)

y_pred_rf=model_rf.predict(X_test)

score_rf, cmt_rf = model_rf.evaluate(y_test, y_pred_rf)

print(f'Accuracy Score for RF:{score_rf}')

print(f"""Confusion Matrix RF:
    {cmt_rf}""")

visualizer.plot_conf_matrix(y_test, y_pred_rf, 'Confusion Matrix in percent for RF', figsize=(10, 8))


model_xgb=Model(model_name_xgb, param_xgb)

model_xgb.train(X_train, y_train)

y_pred_xgb=model_xgb.predict(X_test)

score_xgb, cmt_xgb = model_xgb.evaluate(y_test, y_pred_xgb)

print(f'Accuracy Score for XGB:{score_xgb}')

print(f"""Confusion Matrix XGB:
    {cmt_xgb}""")

visualizer.plot_conf_matrix(y_test, y_pred_xgb, 'Confusion Matrix in percent for XGB', figsize=(10, 8))


models = [model_svc, model_dt, model_rf, model_xgb]
model_names = ['SVC', 'Decision Tree', 'Random Forest', 'XGBoost']

Model.get_evaluation_metrics(models, model_names, X_test, y_test)


# # ===============================================

model_params = {
    'svm': {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC())
        ]),
        'params': {
            'svc__C': [0.1, 1, 10],
            'svc__gamma': ['scale', 'auto'],
            'svc__probability': [True, False],
            'svc__kernel': ['linear', 'rbf'],
            'svc__random_state': [0, 42, 96]

        }
    },
    'random_forest': {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier())
        ]),
        'params': {
            'rf__n_estimators': [i for i in range(50, 400, 50)],            
            'rf__max_depth': [3, 5, 7],
            'rf__min_samples_leaf': [1, 2, 4],
            'rf__min_samples_split': [2, 5, 10],
            'rf__random_state': [0, 42, 96]
        }
    }
}


tune_model=FineTuningModel()

best_estimators_gs, tuned_models_gs = tune_model.tune_models_with_gridsearch(X_train, y_train, model_params)

for estimator in best_estimators_gs:
    print(estimator)

y_pred_rf_gs = tune_model.predict_tuned_model(tuned_models_gs, 'random_forest', X_test)
y_pred_svc_gs = tune_model.predict_tuned_model(tuned_models_gs, 'svm', X_test)



visualizer.plot_conf_matrix(y_test, y_pred_rf_gs, 'Confusion Matrix in percent for RF', figsize=(10, 8))

visualizer.plot_conf_matrix(y_test, y_pred_svc_gs, 'Confusion Matrix in percent for SVC', figsize=(10, 8))


tune_model.evaluate_hpt('Random forest-gs', y_test, y_pred_rf_gs)

tune_model.evaluate_hpt('SVM-gs', y_test, y_pred_svc_gs)

    

best_estimators_rs, tuned_models_rs = tune_model.tune_models_with_randomizedsearch(X_train, y_train, model_params)


for estimator in best_estimators_rs:
    print(estimator)

y_pred_rf_rs = tune_model.predict_tuned_model(tuned_models_rs, 'random_forest', X_test)
y_pred_svc_rs = tune_model.predict_tuned_model(tuned_models_rs, 'svm', X_test)


tune_model.evaluate_hpt('Random forest-rs', y_test, y_pred_rf_rs)

tune_model.evaluate_hpt('SVM-rs', y_test, y_pred_svc_rs)

visualizer.plot_conf_matrix(y_test, y_pred_rf_rs, 'Confusion Matrix in percent for RF', figsize=(10, 8))

visualizer.plot_conf_matrix(y_test, y_pred_svc_rs, 'Confusion Matrix in percent for SVC', figsize=(10, 8))
