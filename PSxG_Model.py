# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from matplotlib.colors import to_rgba
from statsbombpy import sb
from mplsoccer import Pitch, FontManager, VerticalPitch
import seaborn as sns
from matplotlib.cm import get_cmap
from PSxG_Features_Function import *

"""1) Open file with information about competition and matches"""
#Define the path
#path=C:\Users\david\OneDrive\Football Analytics\Calcio\Dati e Progetti\Dati\open-data-master\data

#Open the file
with open(rf'{path}\competitions.json') as data_file:
    #print (mypath+'events/'+file)
    competitions = json.load(data_file)
    
#Convert in a dataframe
competitions_df = pd.json_normalize(competitions, sep="_")


"""2) Creation of a Shot Dataframe to train Models"""

""" 2A) Filtering competition file for the competitions that i want analyze"""

"""Gat only Male competition"""
competitions_df_male = competitions_df[competitions_df['competition_gender'] == 'male']

"""Delate from my analysis oldest competitions (before 2000) because the ball and the athleticism of players were different.
   The idea is to create a group of omogeneus shots (same average speed, effect, ability of GK, ecc.)

For the same reason delate U20 competitions too."""
# List with competitions years to delate.
lista_anni_da_escludere = ['1958', '1962', '1970', '1974', '1986', '1990']
# Filter my competitions file
competitions_df_filt = competitions_df_male[(competitions_df_male['competition_name'] != 'FIFA U20 World Cup') & (
    (competitions_df_male['competition_name'] != 'FIFA U20 World Cup') & (~competitions_df_male['season_name'].isin(lista_anni_da_escludere)))]


"""2B) Iterate over the ids of the various selected competitions to get the event shots."""

"""ID competition extraction"""
competition_id_list = list(competitions_df_filt['competition_id'].unique())

# Define lists to insert shots and matches information.
shots_matches = []
len_match_list = []
numero_match = 0

"""Let's start to iterate different competition"""
for competition_id in competition_id_list:
    
    # Competitions datafrmae filter for season 
    season_id_df = competitions_df_filt[competitions_df_filt['competition_id']
                                        == competition_id]

    """ID season extraction"""
    season_id_list = list(season_id_df['season_id'].unique())

    #Matches extraction for the current season.
    for season_id in season_id_list:
        # competition_id,season_id=9,281
        print(competition_id, season_id)

        # Extraction the information about matches id for that season and competition.
        #season_path=C:\Users\david\OneDrive\Football Analytics\Calcio\Dati e Progetti\Dati\open-data-master\data\matches
        with open(rf'{season_path}\{competition_id}\{season_id}.json',  encoding="utf8") as data_file:
            #print (mypath+'events/'+file)
            data = json.load(data_file)
            # Find number of matches for that season and competition
            lenght = len(data)
            print(f'Match Number:{lenght}')
            numero_match += lenght
            len_match_list.append(lenght)
            # Iteration on the number of matches
            for i in range(lenght):

                # Open every matches.
                d = data[i]

                match_year = int(d['match_date'][:4])

                if match_year > 1999:
                    # Match ID extraction.
                    match_id = d['match_id']
                    
                    #match_path='C:\Users\david\OneDrive\Football Analytics\Calcio\Dati e Progetti\Dati\open-data-master\data\events'
                    
                    with open(rf'{match_path}\{match_id}.json', encoding="utf8") as data_file:

                        Data = json.load(data_file)

                    file_name = str(match_id)+'.json'

                    #Convert Jason in pandas dataframe
                    from pandas import json_normalize
                    df = json_normalize(Data, sep="_").assign(
                        match_id=file_name[:-5])

                    # Exctract only shots events
                    shots = df[df['type_name'] == 'Shot']

                    # Insert my matches shots df inside shots list
                    shots_matches.append(shots)

                else:
                    continue

"""3) CREATION OF A SHOTS DATAFRAME AND A RESULT DATAFRMAE FOR RESULTS."""
from PSxG_Features_Function import *

"""3A) Concat my shots matches dataframe"""
total_shots = pd.concat(shots_matches)
columns=total_shots.columns

#Reset Index of my shots dataframe.
total_shots=total_shots.reset_index(drop=True)

"""3B) RESULTS DATAFRMAE CREATION"""
Result_models_summary=pd.DataFrame(columns=['Log_Loss_train','Log_Loss_test','Brier_Score_train','Brier_Score_test','F1_train','F1_test','AUC_train','AUC_test','Log_Loss_Calibrated_train','Log_Loss_Calibrated_test','Brier_Score_Calibrated_train','Brier_Score_Calibrated_test','F1_Calibrated_train','F1_Calibrated_test','AUC_Calibrated_train','AUC_Calibrated_test'],index=['Baseline','Logistic Regression','Random Forest','XG Boost'])


"""3) I Create a DF to Train model"""

"""Filter fo only on targhet shots"""
OT_Shots=total_shots[(total_shots['shot_outcome_name']=='Goal') | (total_shots['shot_outcome_name']=='Saved') | (total_shots['shot_outcome_name']=='Saved To Post')]

OT_shots_XG=OT_Shots[['shot_statsbomb_xg','shot_outcome_name']]

"""Filter for only open play shots"""
OT_Shots_op=OT_Shots[OT_Shots['shot_type_name']=='Open Play']

"""3) Features creation"""

"""3A) Exctract shots cordinates"""
OT_Shots_op[['location_x', 'location_y','location_z']] = OT_Shots_op.apply(extract_coordinates, axis=1, column_name='location')
OT_Shots_op[['end_location_x', 'end_location_y', 'end_location_z']] = OT_Shots_op.apply(extract_coordinates,axis=1, column_name='shot_end_location')


"""3B) Shot distance calculation"""
OT_Shots_op['shot_distance']=OT_Shots_op.apply(Shot_distance_from_center, axis=1)

"""3C) Shot angle calculation"""
OT_Shots_op['shot_angle']=OT_Shots_op.apply(Shot_angle, axis=1)

"""3D) Shot speed calculation"""
OT_Shots_op['shot_speed'] = OT_Shots_op.apply(Shot_speed, axis=1)

"""3E) Players inside shot cone and cone density calculation"""
OT_Shots_op['players_inside_shot_cone'],OT_Shots_op['cone_density'] = zip(*OT_Shots_op.apply(number_of_defenders_inside_shot_cone, axis=1))

"""3F) Distance from the two closed defenders at the moment of the shot"""
OT_Shots_op['Distance_D1'],OT_Shots_op['Distance_D2'] = zip(*OT_Shots_op.apply(Distance_to_D1_and_D2, axis=1))

"""3G) Distance from GK and the shoter, and position of GK at the moment of Shot calculation."""
OT_Shots_op['GK_distance_to_shoter'],OT_Shots_op['GK_location_x'],OT_Shots_op['GK_location_y'] = zip(*OT_Shots_op.apply(Distace_to_keeper_and_coordinate, axis=1))

"""3H) Distance from GoalKeeper and goal center calculation"""
OT_Shots_op['GK_distance_to_goal_center'] = OT_Shots_op.apply(Distace_to_keeper_and_goal_center, axis=1)

"""3I) Keeper_Angle"""
OT_Shots_op['Keeper_Angle'] = OT_Shots_op.apply(Angle_Keeper_and_Posts, axis=1)
OT_Shots_op['end_location_zone'] = OT_Shots_op['end_location_y'].apply(convert_end_y_location)

"""3L) Get only columns for treining Model."""
OT_Shots_op_features=OT_Shots_op[['shot_distance','shot_angle','shot_speed','players_inside_shot_cone','cone_density','Distance_D1','Distance_D2','GK_distance_to_shoter','GK_distance_to_goal_center','Keeper_Angle', 'end_location_zone', 'end_location_z','shot_body_part_name','shot_technique_name','under_pressure','shot_open_goal','shot_deflected']]

"""3M) Transform categorical variables (strings) into numbers."""
"""I use One-Hot Encoding because the categories are not ordinal and by doing so I do not give the system a hierarchy of importance through values."""

# one-hot encoding on shot body part
OT_Shots_op_features= pd.get_dummies(OT_Shots_op_features, columns=['shot_body_part_name'], prefix='encode', drop_first=True)
# one-hot encoding on shot technique
OT_Shots_op_features= pd.get_dummies(OT_Shots_op_features, columns=['shot_technique_name'], prefix='encode', drop_first=False)
#After Multicollinearity analisis on shot tecniche encoded features, Normal technique are collineary with other, i decided to delate that and not randomly with drop first.
OT_Shots_op_features=OT_Shots_op_features.drop('encode_Normal',axis=1)
# one-hot encoding on end_location_zone
OT_Shots_op_features= pd.get_dummies(OT_Shots_op_features, columns=['end_location_zone'], prefix='encode', drop_first=True)


"""3N) Get only columns for treining Model."""
features_columns=OT_Shots_op_features.columns

"""3O) Label encding for Shot outcome 1=Goal and 0=No Goal"""
# Convertion of column shot outcome.
OT_Shots_op_features['encode_shot_outcome_name'] = OT_Shots_op['shot_outcome_name'].apply(lambda x: 1 if x == 'Goal' else 0)

"""3P) For Boolean columns sobstitute nan value with Boolean values"""
OT_Shots_op_features[['under_pressure', 'shot_open_goal', 'shot_deflected']] = OT_Shots_op_features[['under_pressure', 'shot_open_goal', 'shot_deflected']].fillna(False)

"""3Q) Delate RAW with nan value in GK position."""
OT_Shots_op_features_clean=OT_Shots_op_features.dropna()

"""4) Start with EDA to understand Features-Features and Features-Targhet correlation and for menage Multicollinearity."""

"""4A) EDA."""

"""4B) To check how much the targhets are imbalanced"""

#Calculation number of goal an no goal
targhet_number=OT_Shots_op_features_clean['encode_shot_outcome_name'].value_counts()

#Calculation score goal/no goal
tarhet_score=(targhet_number.loc[1]/(targhet_number.loc[0]+targhet_number.loc[1]))*100


"""4C) Shere numerical an cateorical features"""

#Separate numeric and categorical features
OT_Shots_op_features_clean_numerical=OT_Shots_op_features_clean.select_dtypes(exclude='bool')
OT_Shots_op_features_clean_bool=OT_Shots_op_features_clean.select_dtypes(include='bool')

"""4D) Check if numerical features have outliers
Using BOX PLOT"""
OT_Shots_op_features_clean_numerical.boxplot(figsize=(28, 12))
plt.show()

"""4E) Delate shot_outcome_name from columns to normalize"""
#Save as variable the targhet
targhet=OT_Shots_op_features_clean_numerical['encode_shot_outcome_name']
targhet=pd.DataFrame(targhet)

#Delate Targhet from main df
#Keep a DF with targhet.
OT_Shots_op_features_clean_numerical_with_targhet=OT_Shots_op_features_clean_numerical
#Create DF without targhet.
OT_Shots_op_features_clean_numerical=OT_Shots_op_features_clean_numerical.drop('encode_shot_outcome_name',axis=1)

"""4F) Check correlation from features and Targhet to find possible data Likage"""

#Analysis For Numerical features
#Correlation
correlation_df=Different_Corr(OT_Shots_op_features_clean_numerical,targhet)

#t-test
targhet_name='encode_shot_outcome_name'
t_test_df=t_test(OT_Shots_op_features_clean_numerical_with_targhet,targhet_name)

#Analysis For Bool features
chi2_df=Chi2(OT_Shots_op_features_clean_bool,targhet)

"""4G) detect Correlation from my features using correlation matrix"""
"""Pearson"""
# Correlation Matrix Calculation
correlation_matrix = OT_Shots_op_features_clean.corr()

#Define treshold to filtered correlation
threshold=0.7
correlation_matrix_filtered = correlation_matrix[(correlation_matrix.abs() >= threshold) ]

# Correlation Matrix Visualization
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice di Correlazione Pearson delle Feature")
plt.show()

"""Spearman"""
# Correlation Matrix Calculation
spearman_corr = OT_Shots_op_features_clean.corr(method='spearman')
#Define treshold to filtered correlation
threshold=0.7
spearman_corr_filtered = spearman_corr[(spearman_corr.abs() >= threshold) ]

# Correlation Matrix Visualization
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice di Correlazione di Spearman delle Feature")
plt.show()



"""4H) I try to find multicollinarity with VIF metod"""
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Create df only for VIF
OT_Shots_op_features_clean_vif=OT_Shots_op_features_clean

# convert only boolean columns
bool_columns = OT_Shots_op_features_clean_vif.select_dtypes(include=[bool]).columns
OT_Shots_op_features_clean_vif[bool_columns] = OT_Shots_op_features_clean_vif[bool_columns].astype(int)

# VIF Calculation
vif_data = pd.DataFrame()
vif_data["Feature"] = OT_Shots_op_features_clean_vif.columns
vif_data["VIF"] = [variance_inflation_factor(OT_Shots_op_features_clean_vif.values, i) for i in range(OT_Shots_op_features_clean_vif.shape[1])]

#Define list of columns to remove to delate Multicollinearity
colonne_da_rimuovere=['GK_distance_to_shoter']

#Delate features with hih VIF
OT_Shots_op_features_clean_from_features_vif=OT_Shots_op_features_clean_vif.drop(columns=colonne_da_rimuovere)

# VIF calulation with removed columns
vif_data_clean = pd.DataFrame()
vif_data_clean["Feature"] = OT_Shots_op_features_clean_from_features_vif.columns
vif_data_clean["VIF"] = [variance_inflation_factor(OT_Shots_op_features_clean_from_features_vif.values, i) for i in range(OT_Shots_op_features_clean_from_features_vif.shape[1])]

#Correlation after features delate
correlation_matrix_after_remotion = OT_Shots_op_features_clean_vif.corr()


"""5) Create Train and Test data and apply normalization on my dataset."""
"""5A) Normalization"""

#A) RobustScaler in case presence of outliers in 
from sklearn.preprocessing import StandardScaler

#Apply standard scaler
scaler = StandardScaler()
OT_Shots_op_features_clean_scaled = scaler.fit_transform(OT_Shots_op_features_clean_numerical.select_dtypes(include=[np.number]))

#Convert in df
OT_Shots_op_features_clean_numerical_normalized=pd.DataFrame(OT_Shots_op_features_clean_scaled,columns=OT_Shots_op_features_clean_numerical.select_dtypes(include=[np.number]).columns,index=OT_Shots_op_features_clean_numerical.index)


"""5B) Concat normalized numeircal df and cateorical features.
Creating final Dataframe"""

#Concat df
OT_Shots_op_features_clean_normalized = pd.concat([OT_Shots_op_features_clean_numerical_normalized, OT_Shots_op_features_clean_bool], axis=1)


"""5C) Import model library and ML techinque."""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold,cross_val_score
from sklearn.feature_selection import RFECV


"""5D) Split my dataframe in train and test"""
#Split in Train and test my datast: i Use division 80% Train and 20% test.
X_train, X_test, y_train, y_test = train_test_split(OT_Shots_op_features_clean_normalized, targhet, test_size=0.2, stratify=targhet, random_state=42)

"""5E) Find weight to balance logistic regression for difference in targhet 1 and 0"""
count_class_0, count_class_1 = y_train.value_counts()

weight_0 = count_class_0 / (count_class_0 + count_class_1)
weight_1 = count_class_1 / (count_class_0 + count_class_1)


"""6) Let's Start to create the model"""

"""LOGISTIC REGRESSION"""


"""6A) Define my Model"""
# Define Logistic regression with L2 (Ridge), at this time i don't use L1 (Lasso) because my dataset is very well structured and every features could be rilevant.
#I chose liblinear for solver because is ok for L2 with medium dataframe.
model = LogisticRegression(penalty='l2', solver='liblinear',class_weight={0:weight_0, 1:weight_1})

"""6B) Define cross-Validation"""
#Define Stratified Cross validation (I have imbalanced targhet, this cross validation menage this problem, keep the same score betweente taghets fo every fold) 
#Set 10-fold for cross validation (10-fold is ok for little and medium dataframe like this case, and the samples are bier every fold than 5-fold and the train will be better every time.)
cv = StratifiedKFold(n_splits=10)

"""6C) RECUSIVE FEATURES ELIMINATION"""
"""6C1)  Define Recursive Features Elimination"""
#RFE inside the cross validation find features to eliminate.
#I use log-loss because the idea of this project is to estimate the probability that a shot is gol or not and log-loss is perfect.
#Skilearn use neg_log_loss, it is log_loss but negative, and more close to 0 is the value better is the model: -o.2 is better than -0.5 for neg log_loss
rfecv = RFECV(estimator=model, step=1, cv=cv, scoring='neg_log_loss')  
rfecv.fit(X_train, y_train)

print(f"Optimal number of features: {rfecv.n_features_}")

"""6C2) Plot for every features combination the result in mean of my neg_log_loss.
The idea is to select the number of feature that minimize in mean the log-loss"""
cv_results = pd.DataFrame(rfecv.cv_results_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean neg_log_Loss")
plt.errorbar(
    x=cv_results.index,
    y=cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
)

plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()


"""6C3) Exctract features to remove"""
# Obtain features to eliminate
features_to_remove = [i for i, x in enumerate(rfecv.support_) if not x]
print("Features index:", features_to_remove)

# Obtain features ranking
print("Features ranking:", rfecv.ranking_)

# Obtain features name to remove
columns_to_remove = X_train.columns[~rfecv.support_]
print("Features name to remove:", columns_to_remove.tolist())

"""6D) Cross Validation"""

"""6D1) Try Nasted cross validation"""
#Define parameters values to Test.
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'max_iter': [100, 200, 300,400],
    'penalty': ['l1', 'l2'],
    'class_weight': ['balanced', {0:weight_0, 1:weight_1}]}

#Set Outer cross validation
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

#Set Inner cross validation
cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

#Set model
model=LogisticRegression()

"""6D2) Start with Nested cross Validation"""
#Apply Grid_SerachCv for test optimal parameters values to minimize Log-Loss.
grid = RandomizedSearchCV(model, param_grid, cv=cv_inner, scoring='neg_log_loss', n_jobs=-1)
scores = cross_val_score(grid, X_train, y_train, cv=cv_outer, scoring='neg_log_loss', n_jobs=-1)

#Train on the entaire training dataset
grid.fit(X_train, y_train)

#Print the best parameters and training time
print("Best Hyperparameters: ", grid.best_params_)

"""6D3) Creatin best model with tuned parameters"""
#best_model = grid.best_estimator_
tuned_model_lr= grid.best_estimator_


"""6D4) With the Best Hyperparameters train my model"""
tuned_model_lr.fit(X_train, y_train)


"""7) Testing Model and calculate PSXG"""
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

"""7A) Train"""
#Prediction of PSxG on Train
y_train_pred_proba_lr = tuned_model_lr.predict_proba(X_train)
#Evaluation predictions with Log-Loss on the Train.
train_logloss_lr = log_loss(y_train, y_train_pred_proba_lr)

#Add label and predict probability to my Train Shots
y_train_pred_proba_lr=pd.DataFrame(y_train_pred_proba_lr) 
y_train_pred_proba_lr['real_targhet']=list(y_train['encode_shot_outcome_name'])
y_train_pred_proba_lr['real_ind']=list(y_train.index)

"""7B) Test"""

#Prediction of PSxG on Train
y_test_pred_proba_lr = tuned_model_lr.predict_proba(X_test)
#Evaluation predictions with Log-Loss on the Train.
test_logloss_lr = log_loss(y_test, y_test_pred_proba_lr)

#Add label and predict probability to my Test Shots.
y_test_pred_proba_lr=pd.DataFrame(y_test_pred_proba_lr) 
y_test_pred_proba_lr['real_targhet']=list(y_test['encode_shot_outcome_name'])
y_test_pred_proba_lr['real_ind']=list(y_test.index)

"""7C) Calculate Predict Targhet"""
#Train
y_train_pred_lr = tuned_model_lr.predict(X_train)
#Test
y_test_pred_lr = tuned_model_lr.predict(X_test)



"""8) Create the Bankmark to compare my models"""

"""8A)Bankmark model prediction and Log-Loss evaluation"""

# A. Baseline classe maggioritaria
p_majority = np.mean(y_test == 0)  # Probabilità di classe maggioritaria
log_loss_majority_bankmark = log_loss(y_test, [p_majority] * len(y_test))

# B. Baseline media del target
#Train
p_mean_train = np.mean(y_train)  # Probabilità della classe positiva (media del target)
log_loss_train_mean_bankmark = log_loss(y_train, [p_mean_train] * len(y_train))

#Test
p_mean_test = np.mean(y_test)  # Probabilità della classe positiva (media del target)
log_loss_test_mean_bankmark = log_loss(y_test, [p_mean_test] * len(y_test))

print(f"Log Loss Baseline (Classe maggioritaria): {log_loss_majority_bankmark:.4f}")
print(f"Log Loss Baseline (Media del target): {log_loss_mean_bankmark:.4f}")


"""8B) Brier Score evaluation"""
#Train
brier_train_mean_bankmark = brier_score_loss(y_train, [p_mean_train] * len(y_train))

#Test
brier_test_mean_bankmark = brier_score_loss(y_test, [p_mean_test] * len(y_test))

"""8C) Save random model Log Loss"""
Result_models_summary.loc['Baseline','Log_Loss_train']=log_loss_train_mean_bankmark
Result_models_summary.loc['Baseline','Log_Loss_test']=log_loss_test_mean_bankmark

"""8D) Save random model Brier Score"""
Result_models_summary.loc['Baseline','Brier_Score_train']=brier_train_mean_bankmark
Result_models_summary.loc['Baseline','Brier_Score_test']=brier_test_mean_bankmark


"""9) Isolate Shots with PSxG>=0.90 to understand the pattern that model captured"""

"""9A) Select Shots with probability over 0.9 to score"""
#Select only shots with a certain rane of PSXG
y_prob_high_lr=y_test_pred_proba_lr[(y_test_pred_proba_lr[1]<1) & (y_test_pred_proba_lr[1]>=0.9)]
#et index of selected PSXG range
index_prob_high_lr=list(y_prob_high_lr['real_ind'])

#Filtered df with features of shots in selected PSXG range
high_case_lr=OT_Shots_op_features_clean.loc[index_prob_high_lr]
#Filtered df with features of shots not in selected PSXG range
non_high_case_lr=OT_Shots_op_features_clean.loc[~OT_Shots_op_features_clean.index.isin(list(high_case_lr.index))]

"""10) Model Summary"""
"""Chack how Features impact on Prediction of PSxG."""
from tabulate import tabulate
from scipy.stats import norm

# Get the estimated coefficients
coef = tuned_model_lr.coef_[0]
intercept = tuned_model_lr.intercept_[0]

# Calculate the standard errors
n = len(y_train)
A = np.hstack((np.ones((n, 1)), X_train))
p = len(coef)
y_pred = pd.DataFrame(tuned_model_lr.predict(X_train))
y_train_reset=y_train.reset_index()
residuals = y_train_reset['encode_shot_outcome_name']- y_pred[0]
sigma2 = np.sum(residuals**2) / (n - p - 1)
A = A.astype(float)
cov = sigma2 * np.linalg.inv(np.dot(A.T, A))
se = np.sqrt(np.diag(cov)[1:])

# Calculate the Wald statistics and p-values
wald = coef / se
p_values = (1 - norm.cdf(np.abs(wald))) * 2
features = list(X_train.columns)

# Create a summary table of coefficients, standard errors, Wald statistics, and p-values
table = np.column_stack((features, coef, se, wald, p_values))
headers = ['Feature', 'Coef.', 'Std. Err.', 'Wald', 'p-value']
stats_resume=pd.DataFrame(table,columns=headers)

"""11) Run the Features Importance with the SHAP Values"""
import shap

# Convert only boolean columns to numeric
bool_columns = X_train.select_dtypes(include=[bool]).columns
X_train[bool_columns] = X_train[bool_columns].astype(int)

# SHAP values calculation
explainer_lr = shap.Explainer(tuned_model_lr, X_train)
shap_values_lr = explainer_lr.shap_values(X_train)

#Features Name
feature_names = [a + ": " + str(b) for a,b in zip(X_train.columns, np.abs(shap_values_lr).mean(0).round(2))]

# SHAP Visualizatiopn
plt.figure(figsize=(10, 6))
plt.suptitle("SHAP Summary Plot - Logistic Regressiont", fontsize=14, fontweight="bold")
shap.summary_plot(shap_values_lr, X_train, max_display=X_train.shape[1],feature_names=feature_names)
plt.show()


"""12) Isotonic regulation and Calibration Curve probability"""
from ML_PSxG_Function import *
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

"""12A) Make calibration"""
# Calibrator Initialization
iso_reg = IsotonicRegression(out_of_bounds="clip")  # Clip avoid value out of [0,1]

# Fit the model and get the calibrated probabilities

#Train
y_train_proba_calibrated_lr = iso_reg.fit_transform(y_train_pred_proba_lr[1], y_train_pred_proba_lr['real_targhet'])
y_train_proba_calibrated_lr=pd.DataFrame(y_train_proba_calibrated_lr)
y_train_pred_calibrated_lr = (y_train_proba_calibrated_lr.iloc[:,0] >= 0.5).astype(int)

#Test
y_test_proba_calibrated_lr = iso_reg.fit_transform(y_test_pred_proba_lr[1], y_test_pred_proba_lr['real_targhet'])
y_test_proba_calibrated_lr=pd.DataFrame(y_test_proba_calibrated_lr)
y_test_pred_calibrated_lr = (y_test_proba_calibrated_lr.iloc[:,0] >= 0.5).astype(int)

"""12B) Plot the graph with both curves, those of the uncalibrated model, and the calibrated ones."""
# Before Calibration
fraction_of_positives, mean_predicted_value = calibration_curve(y_test_pred_proba_lr['real_targhet'], y_test_pred_proba_lr[1], n_bins=10)
# After Isotonic Calibration
fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(y_test_pred_proba_lr['real_targhet'], y_test_proba_calibrated_lr, n_bins=10)

# Plot the calibration curve
plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Non calibrato")
plt.plot(mean_predicted_value_cal, fraction_of_positives_cal, "s-", label="Calibrato (Isotonic)")
plt.plot([0, 1], [0, 1], "k--", label="Pexgectly calibrated")  
plt.xlabel("Predicted probability")
plt.ylabel("True frequency")
plt.legend()
plt.title("Calibration Curve\nLogistic Regression")
plt.show()


"""13) Add label and index to test calibrated pred probability"""
"""13A) Add label"""
#Predizioni calibrate train
y_train_proba_calibrated_lr=pd.DataFrame(y_train_proba_calibrated_lr)
y_train_proba_calibrated_lr['real_targhet']=list(y_train['encode_shot_outcome_name'])
y_train_proba_calibrated_lr['real_ind']=list(y_train.index)

# Predizioni calibrate test
y_test_proba_calibrated_lr=pd.DataFrame(y_test_proba_calibrated_lr)
y_test_proba_calibrated_lr['real_targhet']=list(y_test['encode_shot_outcome_name'])
y_test_proba_calibrated_lr['real_ind']=list(y_test.index)

"""13B) Select Shots with probability over 0.9 to score"""
#Select only shots with a certain range of PSXG
y_prob_calibrated_high_lr=y_test_proba_calibrated_lr[(y_test_proba_calibrated_lr[0]<1) & (y_test_proba_calibrated_lr[0]>=0.9)]
#Get index of selected PSXG range
index_prob_calibrated_high_lr=list(y_prob_calibrated_high_lr['real_ind'])

#Filtered df with features of shots in selected PSXG range
high_case_calibrated_lr=OT_Shots_op_features_clean.loc[index_prob_calibrated_high_lr]
#Filtered df with features of shots not in selected PSXG range
non_high_case_calibrated_lr=OT_Shots_op_features_clean.loc[~OT_Shots_op_features_clean.index.isin(list(high_case_calibrated_lr.index))]

"""14) I evaluate the adjusted model using various metrics."""

"""14A) Log Loss"""
train_logloss_lr_regulated = log_loss(y_train, y_train_proba_calibrated_lr.iloc[:,0])
test_logloss_lr_regulated = log_loss(y_test, y_test_proba_calibrated_lr.iloc[:,0])


"""14B) Calibration evaluation using the Brier Score"""
from sklearn.metrics import brier_score_loss
#Brier score non Calibrated model
brier_train_non_calibrated_lr = brier_score_loss(y_train, y_train_pred_proba_lr.iloc[:,1])
brier_test_non_calibrated_lr = brier_score_loss(y_test, y_test_pred_proba_lr.iloc[:,1])
#Brier Score Calibrated model
brier_train_calibrated_lr = brier_score_loss(y_train, y_train_proba_calibrated_lr.iloc[:,0])
brier_test_calibrated_lr = brier_score_loss(y_test, y_test_proba_calibrated_lr.iloc[:,0])

print(f"Brier Score train (Non Calibrated): {brier_train_non_calibrated_lr}")
print(f"Brier Score test (Non Calibrated): {brier_test_non_calibrated_lr}")
print(f"Brier Score train (Calibrated): {brier_train_calibrated_lr}")
print(f"Brier Score test (Calibrated): {brier_test_calibrated_lr}")


"""14C) F1 Score Calculation"""
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, ConfusionMatrixDisplay

#F1 Score Calculation
f1_train_non_calibrated_lr = f1_score(y_train, y_train_pred_lr)
f1_test_non_calibrated_lr = f1_score(y_test, y_test_pred_lr)
# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_test_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr)
disp.plot()
plt.title("Confusion Matrix\nLogistic Regression")
plt.show()

#F1 Score Calibrated
f1_train_calibrated_lr = f1_score(y_train, y_train_pred_calibrated_lr)
f1_test_calibrated_lr = f1_score(y_test, y_test_pred_calibrated_lr)

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_test_pred_calibrated_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr)
disp.plot()
plt.title("Confusion Matrix Calibrated\nLogistic Regression")
plt.show()


"""14D) AUC"""
#Non calibrated
from sklearn.metrics import roc_auc_score, roc_curve

#AUC Calculation
auc_train_lr = roc_auc_score(y_train,  y_train_pred_proba_lr.iloc[:, 1])
auc_test_lr = roc_auc_score(y_test,  y_test_pred_proba_lr.iloc[:, 1])
print(f"AUC: {auc_train_lr}")
print(f"AUC: {auc_test_lr}")

# ROC Curve Visualization
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_test_pred_proba_lr.iloc[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'AUC = {auc_test_lr:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.xlabel('FPR (False Positive Rate)')
plt.ylabel('TPR (True Positive Rate)')
plt.title('Curva ROC Logistic Regression')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#Calibrated

#AUC Calculation
auc_train_calibrated_lr = roc_auc_score(y_train,  y_train_proba_calibrated_lr.iloc[:,0])
auc_test_calibrated_lr = roc_auc_score(y_test,  y_test_proba_calibrated_lr.iloc[:,0])
print(f"AUC: {auc_train_calibrated_lr}")
print(f"AUC: {auc_test_calibrated_lr}")

# ROC Curve Visualization
fpr_calibrated_lr, tpr_calibrated_lr, thresholds_calibrated_lr = roc_curve(y_test, y_test_proba_calibrated_lr.iloc[:, 0])

plt.figure(figsize=(8, 6))
plt.plot(fpr_calibrated_lr, tpr_calibrated_lr, label=f'AUC = {auc_test_calibrated_lr:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.xlabel('FPR (False Positive Rate)')
plt.ylabel('TPR (True Positive Rate)')
plt.title('Curva ROC Logistic Regression Calibrated')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


"""15) FILL RESULTS DATAFRAME WITH Logistic Regression RESULTS"""
#Log Loss
Result_models_summary.loc['Logistic Regression','Log_Loss_train']=train_logloss_lr
Result_models_summary.loc['Logistic Regression','Log_Loss_test']=test_logloss_lr
Result_models_summary.loc['Logistic Regression','Log_Loss_Calibrated_train']=train_logloss_lr_regulated
Result_models_summary.loc['Logistic Regression','Log_Loss_Calibrated_test']=test_logloss_lr_regulated
#Brier Score
Result_models_summary.loc['Logistic Regression','Brier_Score_train']=brier_train_non_calibrated_lr
Result_models_summary.loc['Logistic Regression','Brier_Score_test']=brier_test_non_calibrated_lr
Result_models_summary.loc['Logistic Regression','Brier_Score_Calibrated_train']=brier_train_calibrated_lr
Result_models_summary.loc['Logistic Regression','Brier_Score_Calibrated_test']=brier_test_calibrated_lr
#F1 Score
Result_models_summary.loc['Logistic Regression','F1_train']=f1_train_non_calibrated_lr
Result_models_summary.loc['Logistic Regression','F1_test']=f1_test_non_calibrated_lr
Result_models_summary.loc['Logistic Regression','F1_Calibrated_train']=f1_train_calibrated_lr
Result_models_summary.loc['Logistic Regression','F1_Calibrated_test']=f1_test_calibrated_lr
#AUC
Result_models_summary.loc['Logistic Regression','AUC_train']=auc_train_lr
Result_models_summary.loc['Logistic Regression','AUC_test']=auc_test_lr
Result_models_summary.loc['Logistic Regression','AUC_Calibrated_train']=auc_train_calibrated_lr
Result_models_summary.loc['Logistic Regression','AUC_Calibrated_test']=auc_test_calibrated_lr






"""RANDOM FOREST"""


"""16) Define my Model"""
from sklearn.ensemble import RandomForestClassifier

# Define Logistic regression with L2 (Ridge), at this time i don't use L1 (Lasso) because my dataset is very well structured and every features could be rilevant.
#I chose liblinear for solver because is ok for L2 with medium dataframe.
rf_model = RandomForestClassifier()

"""17) Nasted cross validation"""

"""17A) Define Hyperparameter"""
# Number of features
p = len(OT_Shots_op_features_clean_normalized.columns)

# Dynamic calculation of max_features values
max_features_values = [
    1,                      # Min possible value
    int(np.sqrt(p)),        # Square root of the total number of features
    int(np.log2(p)),        # Base 2 logarithm of the total number of features
    int(p / 2),             # Half of the total number of features
    p                       # All features
]

# Remove any duplicates and sort
max_features_values = sorted(set(max_features_values))

#Define Grid of parameters to test
param_grid_rf = {
    'n_estimators': [100, 200, 500],
    'max_depth': [2, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features':max_features_values,
    'class_weight': [{0:weight_0, 1:weight_1}]}

"""17B) Define element for NCV"""
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_val_score,RandomizedSearchCV

#Set Outer cross validation
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Set Inner cross validation
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

import time

start_time = time.time()


"""17B) Start with Nested cross Validation"""
grid = RandomizedSearchCV(rf_model, param_grid_rf, cv=cv_inner, scoring='neg_log_loss', n_jobs=-1,return_train_score=True)
scores = cross_val_score(grid, X_train, y_train, cv=cv_outer, scoring='neg_log_loss', n_jobs=-1)

#Train on the entaire training dataset
grid.fit(X_train, y_train)
end_time = time.time()

rf_training_time = end_time - start_time

"""17C) Show NCV results"""

# Ottenere i risultati del tuning
results_rf = grid.cv_results_

# Extract the value of log loss for test fold in outer loop in nested cross validation.
mean_test_scores_outer= -scores  # Negative because scoring='neg_log_loss'

# Extract the value of mean log loss for train and test in inner loop during CV calculated for every combination of hyperparameter.
mean_train_scores_inner = -grid.cv_results_['mean_train_score']
mean_test_scores_inner = -grid.cv_results_['mean_test_score']

#Print the best parameters and training time
print("Best parameters: ", grid.best_params_)


"""17D) Creatin best model with tuned parameters"""
tuned_model_rf = grid.best_estimator_
tuned_model_rf.fit(X_train, y_train)

"""18) Testing Model and calculate PSXG"""
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

"""18A) Train"""
#Prediction of PSxG on Train
y_train_pred_proba_rf = tuned_model_rf.predict_proba(X_train)
#Evaluation predictions with Log-Loss on the Train.
train_logloss_rf = log_loss(y_train, y_train_pred_proba_rf)

#Add label and predict probability to my Train Shots
y_train_pred_proba_rf=pd.DataFrame(y_train_pred_proba_rf) 
y_train_pred_proba_rf['real_targhet']=list(y_train['encode_shot_outcome_name'])
y_train_pred_proba_rf['real_ind']=list(y_train.index)

"""18B) Test"""

#Prediction of PSxG on Train
y_test_pred_proba_rf = tuned_model_rf.predict_proba(X_test)
#Evaluation predictions with Log-Loss on the Train.
test_logloss_rf = log_loss(y_test, y_test_pred_proba_rf)

#Add label and predict probability to my Test Shots.
y_test_pred_proba_rf=pd.DataFrame(y_test_pred_proba_rf) 
y_test_pred_proba_rf['real_targhet']=list(y_test['encode_shot_outcome_name'])
y_test_pred_proba_rf['real_ind']=list(y_test.index)


"""19) Select Shots with probability over 0.9 to score"""

#Select only shots with a certain rane of PSXG
y_prob_high_rf=y_test_pred_proba_rf[(y_test_pred_proba_rf[1]<1) & (y_test_pred_proba_rf[1]>=0.9)]
#Set index of selected PSXG range
index_prob_high_rf=list(y_prob_high_rf['real_ind'])

#Filtered df with features of shots in selected PSXG range
high_case_rf=OT_Shots_op_features_clean.loc[index_prob_high_rf]
#Filtered df with features of shots not in selected PSXG range
non_high_case_rf=OT_Shots_op_features_clean.loc[~OT_Shots_op_features_clean.index.isin(list(high_case_rf.index))]

"""20) Calculate Predict Targhet"""
#Train
y_train_pred_rf = tuned_model_rf.predict(X_train)
#Test
y_test_pred_rf = tuned_model_rf.predict(X_test)

"""21) Extract features importances (Between calbrated and not calibrated model,
Features importance don't change)"""
# Extract feature importances
importances = tuned_model_rf.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X_train.columns[i] for i in indices]

# Create plot
plt.figure(figsize=(12,8))

# Create plot title
plt.title("Feature Importance Random Forest")

# Create barplot using seaborn
sns.barplot(x=importances[indices], y=names)

# Add x and y axis labels
plt.xlabel("Importance")
plt.ylabel("Feature")

# Show plot
plt.show()



"""22) Run the Features Importance with the SHAP Values"""
import shap


# Convert only boolean columns to numeric
bool_columns = X_train.select_dtypes(include=[bool]).columns
X_train[bool_columns] = X_train[bool_columns].astype(int)

# SHAP values calculation
explainer_rf = shap.Explainer(tuned_model_rf, X_train)
shap_values_rf = explainer_rf.shap_values(X_train)

#Features Name
feature_names_rf = [a + ": " + str(b) for a,b in zip(X_train.columns, np.abs(shap_values_rf).mean(0).round(2))]

print(f"Feature names length: {len(feature_names_rf)}")
print(f"Number of features in X_train: {X_train.shape[1]}")

# SHAP Visualizatiopn
plt.figure(figsize=(10, 6))
plt.suptitle("SHAP Summary Plot - Random Forest", fontsize=14, fontweight="bold")
shap.summary_plot(shap_values_rf, X_train, max_display=X_train.shape[1],feature_names=feature_names)
plt.show()





"""23) Isotonic regulation and Calibration Curve probability"""
from ML_PSxG_Function import *
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

"""23A) Calibrator inizialization"""
iso_reg = IsotonicRegression(out_of_bounds="clip")  # Clip avoid values out of [0,1]

"""23B) Fit the model and get the calibrated probabilities"""
#Train
y_train_proba_calibrated_rf = iso_reg.fit_transform(y_train_pred_proba_rf[1], y_train_pred_proba_rf['real_targhet'])
y_train_proba_calibrated_rf=pd.DataFrame(y_train_proba_calibrated_rf)
y_train_pred_calibrated_rf = (y_train_proba_calibrated_rf.iloc[:,0] >= 0.5).astype(int)

#Test
y_test_proba_calibrated_rf = iso_reg.fit_transform(y_test_pred_proba_rf[1], y_test_pred_proba_rf['real_targhet'])
y_test_proba_calibrated_rf=pd.DataFrame(y_test_proba_calibrated_rf)
y_test_pred_calibrated_rf = (y_test_proba_calibrated_rf.iloc[:,0] >= 0.5).astype(int)

"""23C) Generate the graph with both curves, those of the uncalibrated model, and the calibrated ones."""
# Non calibrated
fraction_of_positives, mean_predicted_value = calibration_curve(y_test_pred_proba_rf['real_targhet'], y_test_pred_proba_rf[1], n_bins=10)
# After Calibratrion
fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(y_test_pred_proba_rf['real_targhet'], y_test_proba_calibrated_rf, n_bins=10)

# Calibration Curve Visualization
plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Non calibrato")
plt.plot(mean_predicted_value_cal, fraction_of_positives_cal, "s-", label="Calibrato (Isotonic)")
plt.plot([0, 1], [0, 1], "k--", label="Pexgectly calibrated")  
plt.xlabel("Predicted probability")
plt.ylabel("True frequency")
plt.legend()
plt.title("Calibration Curve\nRandom Forest")
plt.show()


"""23D) Add label and index to test calibrated pred probability"""
#Train
y_train_proba_calibrated_rf=pd.DataFrame(y_train_proba_calibrated_rf)
y_train_proba_calibrated_rf['real_targhet']=list(y_train['encode_shot_outcome_name'])
y_train_proba_calibrated_rf['real_ind']=list(y_train.index)

# Test
y_test_proba_calibrated_rf=pd.DataFrame(y_test_proba_calibrated_rf)
y_test_proba_calibrated_rf['real_targhet']=list(y_test['encode_shot_outcome_name'])
y_test_proba_calibrated_rf['real_ind']=list(y_test.index)

"""23E) Select Shots with probability over 0.9 to score"""
#Select only shots with a certain rane of PSXG
y_prob_calibrated_high_rf=y_test_proba_calibrated_rf[(y_test_proba_calibrated_rf[0]<1) & (y_test_proba_calibrated_rf[0]>=0.9)]
#Get index of selected PSXG range
index_prob_calibrated_high_rf=list(y_prob_calibrated_high_rf['real_ind'])

#Filtered df with features of shots in selected PSXG range
high_case_calibrated_rf=OT_Shots_op_features_clean.loc[index_prob_calibrated_high_rf]
#Filtered df with features of shots not in selected PSXG range
non_high_case_calibrated_rf=OT_Shots_op_features_clean.loc[~OT_Shots_op_features_clean.index.isin(list(high_case_calibrated_rf.index))]

"""24) Adjusted model evaluation using various metrics."""

"""24A) Log Loss"""
#Regulated Train
train_logloss_rf_regulated = log_loss(y_train, y_train_proba_calibrated_rf.iloc[:,0])
#Regulated Test
test_logloss_rf_regulated = log_loss(y_test, y_test_proba_calibrated_rf.iloc[:,0])


"""24B) Brier score"""
from sklearn.metrics import brier_score_loss
#Non calibrated
brier_train_non_calibrated_rf = brier_score_loss(y_train, y_train_pred_proba_rf.iloc[:,1])
brier_test_non_calibrated_rf = brier_score_loss(y_test, y_test_pred_proba_rf.iloc[:,1])
#Brier Score Calibrated
brier_train_calibrated_rf = brier_score_loss(y_train, y_train_proba_calibrated_rf.iloc[:,0])
brier_test_calibrated_rf = brier_score_loss(y_test, y_test_proba_calibrated_rf.iloc[:,0])

print(f"Brier Score train (Non Calibrated): {brier_train_non_calibrated_rf}")
print(f"Brier Score test (Non Calibrated): {brier_test_non_calibrated_rf}")
print(f"Brier Score train (Calibrated): {brier_train_calibrated_rf}")
print(f"Brier Score test (Calibrated): {brier_test_calibrated_rf}")


"""24C) F1 Score Calculation"""
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, ConfusionMatrixDisplay

#Non calibrated
#F1 Score Calculation 
f1_train_non_calibrated_rf = f1_score(y_train, y_train_pred_rf)
f1_test_non_calibrated_rf = f1_score(y_test, y_test_pred_rf)
# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_test_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp.plot()
plt.title("Confusion Matrix\nRandom Forest")
plt.show()

#Calibrated
#F1 Score Calibrated
f1_train_calibrated_rf = f1_score(y_train, y_train_pred_calibrated_rf)
f1_test_calibrated_rf = f1_score(y_test, y_test_pred_calibrated_rf)

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_test_pred_calibrated_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp.plot()
plt.title("Confusion Matrix Calibrated\nRandom Forest")
plt.show()


"""24D) AUC"""

from sklearn.metrics import roc_auc_score, roc_curve

#Non calibrated

#AUC Calculation
auc_train_rf = roc_auc_score(y_train,  y_train_pred_proba_rf.iloc[:, 1])
auc_test_rf = roc_auc_score(y_test,  y_test_pred_proba_rf.iloc[:, 1])
print(f"AUC: {auc_train_rf}")
print(f"AUC: {auc_test_rf}")

# ROC curve visualization
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_test_pred_proba_rf.iloc[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'AUC = {auc_test_rf:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
plt.xlabel('FPR (False Positive Rate)')
plt.ylabel('TPR (True Positive Rate)')
plt.title('Curva ROC Random Forest')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#Calibrated

#AUC Calculation
auc_train_calibrated_rf = roc_auc_score(y_train,  y_train_proba_calibrated_rf.iloc[:,0])
auc_test_calibrated_rf = roc_auc_score(y_test,  y_test_proba_calibrated_rf.iloc[:,0])
print(f"AUC: {auc_train_calibrated_rf}")
print(f"AUC: {auc_test_calibrated_rf}")

# ROC curve visualization
fpr_calibrated_rf, tpr_calibrated_rf, thresholds_calibrated_rf = roc_curve(y_test, y_test_proba_calibrated_rf.iloc[:, 0])

plt.figure(figsize=(8, 6))
plt.plot(fpr_calibrated_rf, tpr_calibrated_rf, label=f'AUC = {auc_test_calibrated_rf:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.xlabel('FPR (False Positive Rate)')
plt.ylabel('TPR (True Positive Rate)')
plt.title('Curva ROC Random Forest Calibrated')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


"""25) FILL RESULTS DATAFRAME WITH RANDOM FOREST RESULTS"""
#Log Loss
Result_models_summary.loc['Random Forest','Log_Loss_train']=train_logloss_rf
Result_models_summary.loc['Random Forest','Log_Loss_test']=test_logloss_rf
Result_models_summary.loc['Random Forest','Log_Loss_Calibrated_train']=train_logloss_rf_regulated
Result_models_summary.loc['Random Forest','Log_Loss_Calibrated_test']=test_logloss_rf_regulated
#Brier Score
Result_models_summary.loc['Random Forest','Brier_Score_train']=brier_train_non_calibrated_rf
Result_models_summary.loc['Random Forest','Brier_Score_test']=brier_test_non_calibrated_rf
Result_models_summary.loc['Random Forest','Brier_Score_Calibrated_train']=brier_train_calibrated_rf
Result_models_summary.loc['Random Forest','Brier_Score_Calibrated_test']=brier_test_calibrated_rf
#F1 Score
Result_models_summary.loc['Random Forest','F1_train']=f1_train_non_calibrated_rf
Result_models_summary.loc['Random Forest','F1_test']=f1_test_non_calibrated_rf
Result_models_summary.loc['Random Forest','F1_Calibrated_train']=f1_train_calibrated_rf
Result_models_summary.loc['Random Forest','F1_Calibrated_test']=f1_test_calibrated_rf
#AUC
Result_models_summary.loc['Random Forest','AUC_train']=auc_train_rf
Result_models_summary.loc['Random Forest','AUC_test']=auc_test_rf
Result_models_summary.loc['Random Forest','AUC_Calibrated_train']=auc_train_calibrated_rf
Result_models_summary.loc['Random Forest','AUC_Calibrated_test']=auc_test_calibrated_rf


"""XGBoost"""

"""26) Find weight to balance logistic regression for difference in targhet 1 and 0"""
count_class_0, count_class_1 = y_train.value_counts()

scale_pos_weight = count_class_0 / count_class_1


"""27)Create Model and tuning parameters"""
import xgboost as xgb

# Model definition
xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False,scale_pos_weight=scale_pos_weight)

"""Define parameters to tune"""
# Parameter definition for RandomizedSearch
param_dist_xg= {
    'max_depth': [3, 5, 7, 9, 11],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.5, 1],
    'subsample': [0.5, 0.7, 0.9, 1.0],
    'colsample_bytree': [0.3, 0.5, 0.7, 1.0],
    'lambda': [0, 1, 5, 10],
    'alpha': [0, 1, 5, 10],
    'eta': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500, 1000]
}


"""28) Apply Nested cross validation"""

"""28A) Define element of NCV"""
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_val_score,RandomizedSearchCV

#Set Outer cross validation
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

#Set Inner cross validation
cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

import time

start_time = time.time()

"""28B) Start with Nested cross Validation"""

grid = RandomizedSearchCV(xgb_model, param_dist_xg, cv=cv_inner, scoring='neg_log_loss', n_jobs=-1,return_train_score=True)
scores = cross_val_score(grid, X_train, y_train, cv=cv_outer, scoring='neg_log_loss', n_jobs=-1)

#Train on the entaire training dataset
grid.fit(X_train, y_train)
end_time = time.time()

xg_nasted_tuning_time = end_time - start_time

"""28C) Show NCV results"""

results_xg = grid.cv_results_

# Extract the value of log loss for test fold in outer loop in nested cross validation.
mean_test_scores_outer= -scores  #Negative because scoring='neg_log_loss'

# Extract the value of mean log loss for train and test in inner loop during CV calculated for every combination of hyperparameter.
mean_train_scores_inner = -grid.cv_results_['mean_train_score']
mean_test_scores_inner = -grid.cv_results_['mean_test_score']

#Print the best parameters and training time
print("Best parameters: ", grid.best_params_)


""" Creatin best model with tuned parameters"""
tuned_model_xg = grid.best_estimator_
tuned_model_xg.fit(X_train, y_train)



"""29) Testing Model and calculate PSXG"""
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

"""29A) Train"""
#Prediction of PSxG on Train
y_train_pred_proba_xg = tuned_model_xg.predict_proba(X_train)
#Evaluation predictions with Log-Loss on the Train.
train_logloss_xg = log_loss(y_train, y_train_pred_proba_xg)

#Add label and predict probability to my Train Shots
y_train_pred_proba_xg=pd.DataFrame(y_train_pred_proba_xg) 
y_train_pred_proba_xg['real_targhet']=list(y_train['encode_shot_outcome_name'])
y_train_pred_proba_xg['real_ind']=list(y_train.index)

"""29B) Test"""

#Prediction of PSxG on Train
y_test_pred_proba_xg = tuned_model_xg.predict_proba(X_test)
#Evaluation predictions with Log-Loss on the Train.
test_logloss_xg = log_loss(y_test, y_test_pred_proba_xg)

#Add label and predict probability to my Test Shots.
y_test_pred_proba_xg=pd.DataFrame(y_test_pred_proba_xg) 
y_test_pred_proba_xg['real_targhet']=list(y_test['encode_shot_outcome_name'])
y_test_pred_proba_xg['real_ind']=list(y_test.index)


"""30) Select Shots with probability over 0.9 to score"""

#Select only shots with a certain rane of PSXG
y_prob_high_xg=y_test_pred_proba_xg[(y_test_pred_proba_xg[1]<1) & (y_test_pred_proba_xg[1]>=0.9)]
#Set index of selected PSXG range
index_prob_high_xg=list(y_prob_high_xg['real_ind'])

#Filtered df with features of shots in selected PSXG range
high_case_xg=OT_Shots_op_features_clean.loc[index_prob_high_xg]
#Filtered df with features of shots not in selected PSXG range
non_high_case_xg=OT_Shots_op_features_clean.loc[~OT_Shots_op_features_clean.index.isin(list(high_case_xg.index))]

"""30) Calculate Predict Targhet"""
#Train
y_train_pred_xg = tuned_model_xg.predict(X_train)
#Test
y_test_pred_xg = tuned_model_xg.predict(X_test)

"""31) Extract features importances (Between calbrated and not calibrated model,
Features importance don't change)"""
# Extract feature importances
importances = tuned_model_xg.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X_train.columns[i] for i in indices]

# Create plot
plt.figure(figsize=(12,8))

# Create plot title
plt.title("Feature Importance Random Forest")

# Create barplot using seaborn
sns.barplot(x=importances[indices], y=names)

# Add x and y axis labels
plt.xlabel("Importance")
plt.ylabel("Feature")

# Show plot
plt.show()



"""32) Run the Features Importance with the SHAP Values"""
import shap


# Convert only boolean columns to numeric
bool_columns = X_train.select_dtypes(include=[bool]).columns
X_train[bool_columns] = X_train[bool_columns].astype(int)

# SHAP values calculation
explainer_xg = shap.Explainer(tuned_model_xg, X_train)
shap_values_xg = explainer_xg.shap_values(X_train)

#Features Name
feature_names_xg = [a + ": " + str(b) for a,b in zip(X_train.columns, np.abs(shap_values_xg).mean(0).round(2))]

print(f"Feature names length: {len(feature_names_xg)}")
print(f"Number of features in X_train: {X_train.shape[1]}")

# SHAP Visualizatiopn
plt.figure(figsize=(10, 6))
plt.suptitle("SHAP Summary Plot - Random Forest", fontsize=14, fontweight="bold")
shap.summary_plot(shap_values_xg, X_train, max_display=X_train.shape[1],feature_names=feature_names)
plt.show()


"""33) Isotonic regulation and Calibration Curve probability"""
from ML_PSxG_Function import *
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

"""33A) Calibrator inizialization"""

iso_reg = IsotonicRegression(out_of_bounds="clip")  # Clip avoid vaues out of [0,1]

"""33B) Fit the model and get the calibrated probabilities"""
#Train
y_train_proba_calibrated_xg = iso_reg.fit_transform(y_train_pred_proba_xg[1], y_train_pred_proba_xg['real_targhet'])
y_train_proba_calibrated_xg=pd.DataFrame(y_train_proba_calibrated_xg)
y_train_pred_calibrated_xg = (y_train_proba_calibrated_xg.iloc[:,0] >= 0.5).astype(int)

#Test
y_test_proba_calibrated_xg = iso_reg.fit_transform(y_test_pred_proba_xg[1], y_test_pred_proba_xg['real_targhet'])
y_test_proba_calibrated_xg=pd.DataFrame(y_test_proba_calibrated_xg)
y_test_pred_calibrated_xg = (y_test_proba_calibrated_xg.iloc[:,0] >= 0.5).astype(int)

"""33C) Plot the graph with both curves, those of the uncalibrated model, and the calibrated ones."""
# Non Calibrated
fraction_of_positives, mean_predicted_value = calibration_curve(y_test_pred_proba_xg['real_targhet'], y_test_pred_proba_xg[1], n_bins=10)
# Calibrated
fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(y_test_pred_proba_xg['real_targhet'], y_test_proba_calibrated_xg, n_bins=10)

# Calibration Curve Visualization
plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Non calibrato")
plt.plot(mean_predicted_value_cal, fraction_of_positives_cal, "s-", label="Calibrato (Isotonic)")
plt.plot([0, 1], [0, 1], "k--", label="Pexgectly calibrated") 
plt.xlabel("Predicted probability")
plt.ylabel("True frequency")
plt.legend()
plt.title("Calibration Curve\nXGBoost")
plt.show()


"""34) Add label and index to test calibrated pred probability"""
#Predizioni calibrate train
y_train_proba_calibrated_xg=pd.DataFrame(y_train_proba_calibrated_xg)
y_train_proba_calibrated_xg['real_targhet']=list(y_train['encode_shot_outcome_name'])
y_train_proba_calibrated_xg['real_ind']=list(y_train.index)

# Predizioni calibrate test
y_test_proba_calibrated_xg=pd.DataFrame(y_test_proba_calibrated_xg)
y_test_proba_calibrated_xg['real_targhet']=list(y_test['encode_shot_outcome_name'])
y_test_proba_calibrated_xg['real_ind']=list(y_test.index)

"""35) Select Shots with probability over 0.9 to score"""
#Select only shots with a certain rane of PSXG
y_prob_calibrated_high_xg=y_test_proba_calibrated_xg[(y_test_proba_calibrated_xg[0]<1) & (y_test_proba_calibrated_xg[0]>=0.9)]
#Get index of selected PSXG range
index_prob_calibrated_high_xg=list(y_prob_calibrated_high_xg['real_ind'])

#Filtered df with features of shots in selected PSXG range
high_case_calibrated_xg=OT_Shots_op_features_clean.loc[index_prob_calibrated_high_xg]
#Filtered df with features of shots not in selected PSXG range
non_high_case_calibrated_xg=OT_Shots_op_features_clean.loc[~OT_Shots_op_features_clean.index.isin(list(high_case_calibrated_xg.index))]

"""36) Model evaluation with different metrics"""
"""36A) Log Loss"""
#Regolati
train_logloss_xg_regulated = log_loss(y_train, y_train_proba_calibrated_xg.iloc[:,0])
test_logloss_xg_regulated = log_loss(y_test, y_test_proba_calibrated_xg.iloc[:,0])


"""36B) Brier Score"""
from sklearn.metrics import brier_score_loss
#Non calibrated
brier_train_non_calibrated_xg = brier_score_loss(y_train, y_train_pred_proba_xg.iloc[:,1])
brier_test_non_calibrated_xg = brier_score_loss(y_test, y_test_pred_proba_xg.iloc[:,1])
#Calibrated
brier_train_calibrated_xg = brier_score_loss(y_train, y_train_proba_calibrated_xg.iloc[:,0])
brier_test_calibrated_xg = brier_score_loss(y_test, y_test_proba_calibrated_xg.iloc[:,0])

print(f"Brier Score train (Non Calibrated): {brier_train_non_calibrated_xg}")
print(f"Brier Score test (Non Calibrated): {brier_test_non_calibrated_xg}")
print(f"Brier Score train (Calibrated): {brier_train_calibrated_xg}")
print(f"Brier Score test (Calibrated): {brier_test_calibrated_xg}")


"""36C) F1 Score Calculation"""
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, ConfusionMatrixDisplay

#F1 Score non Calibrated
f1_train_non_calibrated_xg = f1_score(y_train, y_train_pred_xg)
f1_test_non_calibrated_xg = f1_score(y_test, y_test_pred_xg)
# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_test_pred_xg)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp.plot()
plt.title("Confusion Matrix\nXGBoost")
plt.show()

#F1 Score Calibrated
f1_train_calibrated_xg = f1_score(y_train, y_train_pred_calibrated_xg)
f1_test_calibrated_xg = f1_score(y_test, y_test_pred_calibrated_xg)

# Confusion Matrix
cm_xg = confusion_matrix(y_test, y_test_pred_calibrated_xg)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_xg)
disp.plot()
plt.title("Confusion Matrix Calibrated\nXGBoost")
plt.show()


"""36D) AUC"""

from sklearn.metrics import roc_auc_score, roc_curve

#Non calibrated

#AUC Calculation 
auc_train_xg = roc_auc_score(y_train,  y_train_pred_proba_xg.iloc[:, 1])
auc_test_xg = roc_auc_score(y_test,  y_test_pred_proba_xg.iloc[:, 1])
print(f"AUC: {auc_train_xg}")
print(f"AUC: {auc_test_xg}")

# ROC Curve Visualization
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, y_test_pred_proba_xg.iloc[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_xg, tpr_xg, label=f'AUC = {auc_test_xg:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
plt.xlabel('FPR (False Positive Rate)')
plt.ylabel('TPR (True Positive Rate)')
plt.title('Curva ROC XGBoost')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#Calibrated

# AUC calculation
auc_train_calibrated_xg = roc_auc_score(y_train,  y_train_proba_calibrated_xg.iloc[:,0])
auc_test_calibrated_xg = roc_auc_score(y_test,  y_test_proba_calibrated_xg.iloc[:,0])
print(f"AUC: {auc_train_calibrated_xg}")
print(f"AUC: {auc_test_calibrated_xg}")

# ROC Visualization
fpr_calibrated_xg, tpr_calibrated_xg, thresholds_calibrated_xg = roc_curve(y_test, y_test_proba_calibrated_xg.iloc[:, 0])

plt.figure(figsize=(8, 6))
plt.plot(fpr_calibrated_xg, tpr_calibrated_xg, label=f'AUC = {auc_test_calibrated_xg:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.xlabel('FPR (False Positive Rate)')
plt.ylabel('TPR (True Positive Rate)')
plt.title('Curva ROC XGBoost Calibrated')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


"""37) FILL RESULTS DATAFRAME WITH XG Boost RESULTS"""
#Log Loss
Result_models_summary.loc['XG Boost','Log_Loss_train']=train_logloss_xg
Result_models_summary.loc['XG Boost','Log_Loss_test']=test_logloss_xg
Result_models_summary.loc['XG Boost','Log_Loss_Calibrated_train']=train_logloss_xg_regulated
Result_models_summary.loc['XG Boost','Log_Loss_Calibrated_test']=test_logloss_xg_regulated
#Brier Score
Result_models_summary.loc['XG Boost','Brier_Score_train']=brier_train_non_calibrated_xg
Result_models_summary.loc['XG Boost','Brier_Score_test']=brier_test_non_calibrated_xg
Result_models_summary.loc['XG Boost','Brier_Score_Calibrated_train']=brier_train_calibrated_xg
Result_models_summary.loc['XG Boost','Brier_Score_Calibrated_test']=brier_test_calibrated_xg
#F1 Score
Result_models_summary.loc['XG Boost','F1_train']=f1_train_non_calibrated_xg
Result_models_summary.loc['XG Boost','F1_test']=f1_test_non_calibrated_xg
Result_models_summary.loc['XG Boost','F1_Calibrated_train']=f1_train_calibrated_xg
Result_models_summary.loc['XG Boost','F1_Calibrated_test']=f1_test_calibrated_xg
#AUC
Result_models_summary.loc['XG Boost','AUC_train']=auc_train_xg
Result_models_summary.loc['XG Boost','AUC_test']=auc_test_xg
Result_models_summary.loc['XG Boost','AUC_Calibrated_train']=auc_train_calibrated_xg
Result_models_summary.loc['XG Boost','AUC_Calibrated_test']=auc_test_calibrated_xg

 
"""38) SAVE THE RESULTS MODEL SUMMARY"""
#path='C:\Users\david\OneDrive\Football Analytics\Calcio\Dati e Progetti\Miei Progetti\Progetto PsXG\Risultati'
Result_models_summary.to_excel(rf'{path}\Result_models_summary.xlsx')

#Get only column i need for report
Result_models_summary_useful=Result_models_summary[['Log_Loss_train','Log_Loss_test','Brier_Score_train','Brier_Score_test','Log_Loss_Calibrated_train','Log_Loss_Calibrated_test','Brier_Score_Calibrated_train','Brier_Score_Calibrated_test']]


