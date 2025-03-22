# -*- coding: utf-8 -*-

"""FUNCTIONS FOR DFFERENT APPLICATION IN ML PROBLEM."""

import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold,cross_val_score
from scipy.stats import pearsonr, spearmanr, pointbiserialr,ttest_ind
from scipy.stats import chi2_contingency


"""1) Calibration Curve"""
def Calibration_Curve(y_test,y_test_pred_proba_rf,model,text):
    
    prob_true_non_cal, prob_pred_non_cal = calibration_curve(y_test, y_test_pred_proba_rf.iloc[:,1], n_bins=10)

    #Plot of non calibrated probability
    plt.plot(prob_true_non_cal, prob_pred_non_cal, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Predicted probability')
    plt.ylabel('True frequency')
    plt.title(f'Calibration curve {text}')
    
    return plt




"""2) Isotonic Regulation"""
def Isotonic_regulation(X_train, y_train,model):
    
    """Apply Isotonic regulation"""

    X_train_split, X_calib, y_train_split, y_calib = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # Calibration with isotonic regression
    calibrated_rf = CalibratedClassifierCV(base_estimator=model, method='isotonic', cv=5)
    calibrated_rf.fit(X_calib, y_calib)
        
    return calibrated_rf



"""3)Function to evaluate relationship between features and target."""

"""3A) Technique for Numerical features"""

"""3A1) Functions for Person, Spearman and Point-Biserial correlation between Numerical and Target features."""

def Different_Corr(df,targhet):
    # Dictionary Creation
    correlations = {
        'Feature': [],
        'Pearson': [],
        'Spearman': [],
        'Point-Biserial': []
    }
    
    numerical_features=df.columns
    
    #Calculations of every type of correlation for every Features
    for numerical_feature in numerical_features:
        print(df[numerical_features].dtypes)
        
        # Pearson Correlation
        correlation_pearson, _ = pearsonr(df[numerical_feature], targhet['encode_shot_outcome_name'])
        
        # Spearman Correlation
        correlation_spearman, _ = spearmanr(df[numerical_feature], targhet['encode_shot_outcome_name'])
        
        # Point-Biserial Correlation
        correlation_pointbiserial, _ = pointbiserialr(df[numerical_feature], targhet['encode_shot_outcome_name'])
        
        # Append results in Dict
        correlations['Feature'].append(numerical_feature)
        correlations['Pearson'].append(correlation_pearson)
        correlations['Spearman'].append(correlation_spearman)
        correlations['Point-Biserial'].append(correlation_pointbiserial)

    # Convert in Pandas Dataframe
    correlation_df = pd.DataFrame(correlations)
    
    return correlation_df

"""3A2) T-test"""

def t_test(df,targhet_name):
    
    #Dictionary Creation
    t_test = {
        'Feature': [],
        't-stat': [],
        'P-Value': []

    }
    
    numerical_features=df.columns
    numerical_features=numerical_features[:-1]
    
    #T-Test calculation for the two different group.
    for numerical_feature in numerical_features:

        # t-test
        group_0 = df[ df[f'{targhet_name}'] == 0][numerical_feature]
        group_1 = df[ df[f'{targhet_name}'] == 1][numerical_feature]
        t_stat, p_value = ttest_ind(group_0, group_1)
        
        # Append results in Dictionary
        t_test['Feature'].append(numerical_feature)
        t_test['t-stat'].append(t_stat)
        t_test['P-Value'].append(p_value)
        
    # Convert in Pandas Dataframe
    t_test_df=pd.DataFrame(t_test)
    
    return t_test_df



"""2) Technique for Categorical features"""

""" CHI Square Test"""
def Chi2(df,targhet):
    
    #Dict Creation
    chi2_dict = {
        'Feature': [],
        'chi2': [],
        'P-Value': []

    }
    
    bool_features=df.columns
    
    #Chi 2 calculation
    for bool_feature in bool_features:
            
        contingency_table = pd.crosstab(df[bool_feature], targhet['encode_shot_outcome_name'])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        # Append results to Dict
        chi2_dict['Feature'].append(bool_feature)
        chi2_dict['chi2'].append(chi2)
        chi2_dict['P-Value'].append(p_value)

    # Convert in Pandas Dataframe
    chi2_df=pd.DataFrame(chi2_dict)
    
    return chi2_df
        