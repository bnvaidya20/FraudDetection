import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.model_selection import GridSearchCV , RandomizedSearchCV, StratifiedKFold


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        data = pd.read_csv(self.filepath)
        return data

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def set_index(self, column_name):
        self.df.set_index(column_name, inplace=True)
    
    def drop_columns(self, column_name):
        return self.df.drop(column_name, axis=1)
    
    def rename_columns(self, column_name):
        self.df=self.df.rename(columns=column_name)
        return self.df

    def check_missing_values(self):
        return print(self.df.isna().sum())

    def handle_missing_values(self, method='ffill'):
        if method=='ffill':
            self.df.ffill(inplace=True)
        elif method == 'bfill':
            self.df.bfill(inplace=True)
        elif method == 'dropna':
            self.df.dropna(inplace=True)
        else:
            print('Not found')

class EmailAnalysisMixin:
    @staticmethod
    def find_uncommon_emails(df, df1):
        emails_df = set(df['customerEmail'].tolist())
        emails_df1 = set(df1['customerEmail'].tolist())
        return emails_df - emails_df1  # Set difference

    @staticmethod
    def find_common_emails(df, df1):
        emails_df = set(df['customerEmail'].tolist())
        emails_df1 = set(df1['customerEmail'].tolist())
        return emails_df & emails_df1  # Set intersection

class DataAnalysis(EmailAnalysisMixin):
    def __init__(self, df):
        self.df=df

    def count_unique_col(self, column_name):
        return self.df[column_name].nunique()
    
    def count_items(self, column_name):
        items_list = self.df[column_name].tolist()
        item_counts = Counter(items_list)
        return item_counts
    
    def get_unique(self, column_name):
        return self.df[column_name].unique()
    
    def get_multi_occurance(self, column_name):
        return self.df[self.df[column_name]>0]


class DataVisualizer:
    @staticmethod
    def plot_histogram(data):
        for column in data.columns:
            plt.figure(figsize=(10, 4))
            sns.histplot(data[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.show()

    @staticmethod
    def plot_distribution_individual_features(data):
    # Distribution of individual features
        fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(20, 15))
        fig.subplots_adjust(hspace=0.5)
        fig.suptitle('Distributions of Features')
        for ax, feature in zip(axes.flatten(), data.columns):
            sns.histplot(data[feature], kde=True, ax=ax)
            ax.set_title(feature)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(data):
        plt.figure(figsize=(12, 10))
        sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Feature Correlation Heatmap")
        plt.show()

    @staticmethod
    def plot_boxplot(data, column, title):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data[column])
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_count(data, column, title, hue=None, figsize=(12, 5)):
        plt.figure(figsize=figsize)
        sns.countplot(x=column, hue=hue, data=data)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    # Scatter plots for bivariate analysis of select features
    def plot_scatterplot(data, x, y, title, hue=None, figsize=(12, 5)):
        plt.figure(figsize=figsize)
        sns.scatterplot(x=x, y=y, hue=hue, data=data)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_boxplot_target(data, x, y, title, figsize=(12, 5)):
        plt.figure(figsize=figsize)
        # Box plot for a numerical feature across the binary target variable
        sns.boxplot(x=x, y=y, data=data)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_barplot_trn(data, x, y, title, hue=None, figsize=(12, 5)):
        plt.figure(figsize=figsize)
        sns.barplot(x=x, y=y, hue=hue, data=data)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_conf_matrix(y_test, y_pred, title, figsize=(12, 5)):
        cmt= confusion_matrix(y_test, y_pred)
        cmt_percent = cmt / np.sum(cmt) * 100
        labels= ["0", "1"] 
        plt.figure(figsize=figsize)
        sns.heatmap(cmt_percent, annot=True, fmt=".2f", cmap="Reds", cbar=False,
            xticklabels=[f'Predicted {label}' for label in labels],
            yticklabels=[f'Actual {label}' for label in labels])      
        plt.title(title)
        plt.show()      

    
class FeatureEngineering:
    def __init__(self, df_base, df_comparison, base_key, comparison_key):
        self.df_base=df_base
        self.df_comparison=df_comparison
        self.base_key=base_key
        self.comparison_key=comparison_key

    def get_aggregate_values(self, value_column=None, condition=None, aggregate='sum'):
        result = []
        for base_value in self.df_base[self.base_key]:
            filtered_df = self.df_comparison[self.df_comparison[self.comparison_key] == base_value]
            if condition is not None:
                filtered_df = condition(filtered_df)
            if value_column is not None:
                if aggregate == 'sum':
                    total = filtered_df[value_column].sum()
                elif aggregate == 'avg':
                    total = filtered_df[value_column].mean()
                elif aggregate == 'max':
                    total = filtered_df[value_column].max()
                else:
                    raise ValueError(f"Unsupported aggregate type: {aggregate}")
            else:
                total = len(filtered_df)
            result.append(total)
        return result
    
    def add_feature_column(self, column_name, category):
        # Filter df_comparison for the specific category
        filtered_df = self.df_comparison[self.df_comparison[column_name] == category]

        # Group by the comparison_key and count the occurrences
        counts = filtered_df.groupby(self.comparison_key).size()

        # Map the counts back to df_base using the base_key
        result = self.df_base[self.base_key].map(counts).fillna(0).astype(int)

        return result

    def count_duplicates(self, df, column_name):
        return df[column_name].map(df[column_name].value_counts() - 1)

class ModelUtils:
    def __init__(self, features, target):
        self.features=features
        self.target=target
        

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

class ModelSelector:
    def __init__(self):
        self.model = None

    def select_classifier(self, model_name, params={}):
        if model_name.lower() == 'randomforest':  
            self.model = RandomForestClassifier(**params)

        elif model_name.lower() == 'svc':
            self.model = SVC(**params)

        elif model_name.lower() == 'decisiontree':
            self.model = DecisionTreeClassifier(**params)

        elif model_name.lower() == 'xgboost':
            self.model = xgb.XGBClassifier(**params)

        else:
            raise ValueError(f"Model '{model_name}' is not supported.")
        
        return self.model
    

class Model:
    def __init__(self, model_name=None, model_params={}):
        """Initialize the Model with a specific classifier and parameters."""
        self.selector = ModelSelector()  
        self.model = self.selector.select_classifier(model_name, model_params)
        self.model_name = model_name.lower()

        # Define your preprocessing steps and model
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),  
            ('model', self.model)
        ])

    def train(self, X_train, y_train):
        """Train the model on the training data."""
        self.pipeline.fit(X_train, y_train)
        
    def predict(self, X_test):

        y_pred = self.pipeline.predict(X_test)

        return y_pred
    
    def evaluate(self, y_test, y_pred):
        # Calculate the metrics of the predictions

        score = accuracy_score(y_test, y_pred)

        cmt= confusion_matrix(y_test, y_pred)

        return score, cmt
    
    # Function to calculate metrics
    def evaluate_model(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc=roc_auc_score(y_test, y_pred)
        return accuracy, precision, recall, f1, roc_auc
    
    @staticmethod
    def get_evaluation_metrics(models, model_names, X_test, y_test):
        for model, name in zip(models, model_names):
            accuracy, precision, recall, f1, roc_auc = model.evaluate_model(X_test, y_test)
            print(f"""{name}: 
                Accuracy: {accuracy:.2f}, 
                Precision: {precision:.2f}, 
                Recall: {recall:.2f}, 
                F1-Score: {f1:.2f},
                ROC AUC: {roc_auc:.2f}""")


class FineTuningModel():
    def __init__(self):
        pass

    def tune_models_with_gridsearch(self, X_train, y_train, model_params):
        best_estimators = []
        tuned_models_dict = {}  # Initialize an empty dictionary to store GridSearchCV instances

        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for model_name, mp in model_params.items():
            clf = GridSearchCV(mp['model'], mp['params'], cv=stratified_kfold, scoring='accuracy', n_jobs=-1)
            clf.fit(X_train, y_train)
            best_estimators.append({
                'model_name': model_name,
                'best_score': clf.best_score_,
                'best_params': clf.best_params_
            })
            tuned_models_dict[model_name] = clf  # Store the GridSearchCV instance

        return best_estimators, tuned_models_dict
 

    def tune_models_with_randomizedsearch(self, X_train, y_train, model_params):
        best_estimators = []
        tuned_models_dict = {}  # Initialize an empty dictionary to store RandomizedSearchCV instances

        stratified_kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

        for model_name, mp in model_params.items():
            clf = RandomizedSearchCV(mp['model'], mp['params'], cv=stratified_kfold, scoring='accuracy', n_jobs=-1)
            clf.fit(X_train, y_train)
            best_estimators.append({
                'model_name': model_name,
                'best_score': clf.best_score_,
                'best_params': clf.best_params_
            })
            tuned_models_dict[model_name] = clf  # Store the RandomizedSearchCV instance

        return best_estimators, tuned_models_dict


    def predict_tuned_model(self, models, model_name, X_test):
        if model_name in models:
            clf = models[model_name]  # Get the GridSearchCV instance for the specified model
            y_pred = clf.predict(X_test)  # Use the best estimator automatically for prediction
            return y_pred
        else:
            print(f"Model '{model_name}' not found.")
            return None

    def evaluate_hpt(self, model_name, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc=roc_auc_score(y_test, y_pred)            
        print(f"""{model_name}
            Accuracy: {accuracy:.2f}, 
            Precision: {precision:.2f}, 
            Recall: {recall:.2f}, 
            F1-Score: {f1:.2f},
            ROC AUC: {roc_auc:.2f}""")

  

 






