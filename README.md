# Post-Shots-xG-Model
**Purpose of the project**: Creation a simple Post-Shots xG model using Free Statsbomb Event Data.

## What are xG and PSxG?:
Expected Goals (xG) are the most famous metrics in football and their purpose is to quantify the probability that a shot will result in a goal before the shot is taken. 

xG are built using thousands shots recorded by different matches of different championships in different season and give and estimates the probability of the goal occurring on a scale between 0 and 1.

There are several more or less complex expected goals models, but in general the information that is used to train the model is the shot angle, the distance from the center of the goal, the amount of players in the shooting cone, the type of action that precedes the 
shot, type of assit, etc.

**xG advantages**:

• Measures the quality of chances created, regardless of the skill of the shooter.

• Helps assess whether a team is scoring more or less than it "should".

• Useful for analyzing the effectiveness of an attack and the ability of a team to create dangerous chances.

**The limitation of xG**:

• Does not take into account how the shot is executed (power, accuracy, spin).

• Does not consider the goalkeeper's ability to save.

• Can underestimate long shots from players with a very good shot.

In practice it does not say how good or bad a player was at shooting at goal or a keeper to save a goal.

To overcome this problem, Post-Shots xG (PSxG) were created.

This model is trained using only the information that is available after the ball has been kicked, excluding all that is prior to the shot.

Some of the information take in account are the shot angle, the distance from the center of the goal, the amount of players in the shooting cone, shot speed, final position of shot, GoalKeeper position, etc.

In practice PSxG measures the probability of a shot ending in a goal after it has been kicked.

P.N. Since the model evaluates the probability that a shot will result in a goal while also knowing the final information about the shot, only shots that are on target are considered in these models because a shot that is not directed towards the goal has by definition a probability of 0 of resulting in a goal.

**PsXG advantages**:

• Evaluates the actual quality of the shot, not just the position.

• Useful for analyzing the skills of shooters (for example who shoots better than average).

• Used to evaluate the performance of goalkeepers, comparing goals conceded with PSxG.

**The limitation of PSxG**:

• It depends on the shot already taken, so it does not help to evaluate the quality of the creation of the chance.

• It does not distinguish between a shot forced by a defender and a well-executed shot in favorable conditions.

• It does not help to predict future performances as much as xG (a nice shot into the top corner can be a random event).

## Step for creating the model:

1) **Collecting the Data**:
   
   Statsbomb relesead during years a lot of Events Data of different matches.
   
   I collected all the data from all the competitions released by Statsbomb.
   
   I then took only the shooting events from their datasets and of these I took only those from the men's competitions excluding the competitions prior to 2000 to create a dataset of shots as similar as possible.
   
   Of these shots I took only those that occurred in Open Play and that were on target so that they had turned into a goal or had been saved by the goalkeeper.
   
   At the end of this proces i got a dataframe of **23097** shots.
   
2) **Features creation**:
   
   Once I aggregated the shots that could be used to create the PSxG model, I calculated and created the features that described the different shots.
   
   As anticipated, I excluded all the features that could describe events prior to the start of the shot.
   
   I therefore used only features that described the shot from when the ball was kicked to when the shot was stopped or turned into a goal.
   
   Below is the list of calculated features:
   
   • **shot_distance**: Distance, in yards, between shot location and goal center.
   
   • **shot_angle**: Angle, in degrees, between shot location, left post and right post.
   
   • **shot_speed**: The average velocity of the shot in yards per second.
   
   • **player_inside_shot_come**: number of Number of players present in the conical area between the shot and goal posts.
   
   • **cone_density**: Aggregate inverse distance for each player behind the ball in the shot cone.
   
   • **Distance_D1 (D2)**: Distance, in yards, between shooter and nearest (second nearest) defender.

   • **GK_distance_to_shoter**: Distance, in yards, between the goalkeeper and the Shooter center.
   
   • **GK_distance_to_goal_center**: Distance, in yards, between the goalkeeper and the goal center.
   
   • **Keeper_Angle**: Angle, in degrees, made by goalkeeper, left post and right post.
   
   • **under_pressure**: StatsBomb measure indicating whether a defender pressured the shot taker.
   
   • **shot_open_goal**: StatsBomb measure indicating whether the shot was in front of an open goal.
   
   • **shot_deflected**: StatsBomb measure indicating whether the shot was deflected by an other player.
   
   • **Shot Technique Type**: Statsbomb defines how the shot was executed in these ways (half volley, volley, overhead, lob, backheel, header, and normal), using One Hot Encoding Technique every each type has become a boolean column (normal columns is delated to get 
     around 
     the dummy trap).
   
   • **Shot Body Part Type**: Statsbomb defines the part of the body with which the shot was made in these ways (Left Foot, Other, Right Foot, Head), using One Hot Encoding Technique every each type has become a boolean column (Head columns is delated to get around 
     the dummy trap).
   
   • **end_location_y**: Measure of horizontal shot location for on-target shots.

     Since it was highly correlated with other features and was too predictive for the final result leading to overfitting, it was divided into 6 different features, making it less granular.
   
     6 new boolean features were then created (Near_to_left_post, Near_to_right_post, Outer_part_of_the_left_post, Outer_part_of_the_right_post, Right_central_part, Left_central_part).
   
     The goal was divided into 6 sections of a size defined along the y-axis and if the final y-value of the shot fell within that section, it was assigned with a True value to the shot.
   
     The Left_central_part features, again for the Dummy trap problem, was eliminated.
   
   • **end_location_z**: Measure of vertical shot location for on-target shots.

4) **Creation of target variable**:
   
   The Targhet variable was created converting the description of the final result of the shot into **0** if it was a shot blocked by the goalkeeper and into **1** if it was converted into a Goal.
 
5) **Model Selection**:
   
   Since we are dealing with a binary classification problem, the models that are best suited to this type of problem are the following classifiers:
   
   • **Logistic Regression**:
      
     Logistic regression is a linear classification model.
   
     It is based on the sigmoid function which predicts the probability that a given example belongs to a given class in our case 1 (Goal) or 0 (No Goal).
   
     ![Screenshot 2025-03-10 234116](https://github.com/user-attachments/assets/9064c400-e20c-4e5d-b716-a24842bdb194)


     • P(Goal) = Probability of the shot becoming a goal.
   
     • e = Euler’s number (~2.718).
   
     • β₀ = The intercept (a baseline value).
   
     • βᵢ = Weights assigned to different shot factors.
   
     • xᵢ = The different shot-related variables (e.g., distance, angle).

   • **Random Forest**:

     ![Screenshot 2025-03-10 234421](https://github.com/user-attachments/assets/e0ee6568-5c46-4779-82c2-3fc4fe836522)


      The random forest is an ensemble model, that is, it combines different models (in this case, multiple decision trees), thus increasing accuracy and decreasing the risk of overfitting.

      Each tree is trained on a subset of the dataset to decrease variance and only on a subset of features, so as to prevent some trees from becoming dominant.
   
      Finally, the class prediction is obtained with the majority vote of the trees.

   • **XGBoost**:
   
    ![Screenshot 2025-03-13 190450](https://github.com/user-attachments/assets/91152e67-9c6d-41b1-83bd-723e17018f3a)


      Unlike the random forest that creates independent trees (Bagging) it creates trees in sequence (Boosting).
   
      In practice each new tree corrects the errors made by the previous one.
   
      The trees are trained to minimize the residual error using the gradient of the loss function.
   
      XGBoost adds L1 and L2 regularizations to minimize the possibility of overfitting and can also automatically handle missing values ​​in the dataset.

7) **Metric Selection**

     **A) Log-Loss**
   
      ![Screenshot 2025-03-15 192024](https://github.com/user-attachments/assets/cfe15105-4554-4259-bab7-0da2a7d1a2ae)

   
      • y<sub>i</sub> = is the actual result (1 = goal, 0 = no goal).
   
      • p<sub>i</sub> =is the probability predicted by the model.
   
      • N = Number of shots.

      **What does it measure?**

      Unlike other metrics, which evaluate the ability to distinguish between goals and non-goals, Log-Loss evaluates how close the predicted probabilities are to the actual outcomes (goal or no goal).

      It also severely penalizes predictions that are highly confident but incorrect.
      
      If a model assigns very high probabilities to an event that does not happen, or very low probabilities to an event that does happen, the Log Loss explodes and the model is severely penalized.
      
      **Why is it useful for xG and PSxG?**

      • Measures the calibration of probabilities: If a model says that a shot has a 70% chance of being a goal, then 7 times out of 10 that shot should really become a goal.
      
      • Distinguishes between “more or less correct” predictions: Saying 0.8 instead of 0.7 is less serious than saying 0.9 instead of 0.1.
      
      • Sensitive to serious errors: If an impossible shot has xG = 0.95, Log Loss punishes it severely.

     **B) Brier Score**

      ![Screenshot 2025-03-15 193353](https://github.com/user-attachments/assets/9a2dbf74-a8ef-4f4c-aa2c-10ec87108b43)

      • p<sub>i</sub> = is the predicted probability.

      • y<sub>i</sub> = is the actual outcome (1 or 0).

      **What does it measure?**
  
      • The Brier Score measures the root mean square error between predicted probabilities and actual outcomes.
      
      • It measures both how close probabilities are to the correct values ​​(accuracy) and how well calibrated they are.

      **Why is it useful for xG and PSxG?**

      • More interpretable than Log Loss: Values ​​close to 0 mean good predictions.
      
      • Does not penalize large errors excessively: If you give 0.99 to a bad shot, the error is 0.9801, while in Log Loss it would be huge.
      
      • Does not distinguish between high/bad and low/bad probabilities: Saying 0.4 instead of 0.8 is penalized as much as saying 0.8 instead of 0.4.

     **P.S.** The combination of these two metrics **(Log-Loss + Brier Score)** is the best way to check the xG model quality.

8) **Models Training and Result**:

    Now that I have created my shot dataset, containing 23097 sample shots and 28 features, and selected the models to test and the metrics to use to evaluate them, I can start training.
   
    **The first step was to perform EDA on the data.**

    **1) Check relations between Features and Target variable**
    It was checked whether the features correlated highly with the target variable to evaluate whether some could be too predictive on the result leading to data likage and therefore to overfitting of the model.
   
    The features most at risk were those related to the final position of the shot that could contain information on the outcome of the shot.
   
    To do this, a spearman correlation and a t-test were performed between the numerical features and the target and a Chi square test (Chi2) between the categorical features and the target.

    **2) Check correlation between Features and VIF values calculation**
   
    Subsequently, it was checked if there were any linear and non-linear correlations between the features that could lead to redundancy of information, furthermore the VIF values ​​were calculated to evaluate if multicollinearity was present.

    From the analyses it was seen that the two highest correlations were between GK_distance_to_shoter and shot_distance and between Keeper_angle and GK_distance_to_goal_center.
   
    From the VIF values ​​it resulted that GK_distance_to_shoter and shot_end_location_y were the features with the highest VIF values.
   
    It was therefore decided to eliminate GK_distance_to_shoter which was both redundant with shot_distance and had a high VIF, and to transform end_shot_location_y into 6 categorical features as previously described, it was not possible to eliminate it as it was a 
    highly relevant feature for the PSxG model.
   
    This significantly lowered the VIF values, no feature exceeded 20 as a value.
   
    For domain reasons and to not make the model too simple, we chose to keep both GK_distance_to_shoter and Keeper_angle, both considered fundamental features.

    **The second step was to training and evaluate.**
   
    **3) Model Training and Evaluation**

    With the clean and adjusted dataset of shots we were ready to train the model.

    The dataset was divided into train and test using the 80-20 rule having a training dataset equivalent to 80% of the original one and the test equivalent to 20%.

    Weights were calculated to give to the model to manage the imbalance of the classes in the target variable.

    The 3 chosen models were trained using the nested cross validation with k-fold = 10 and evaluated using the chosen metrics (Log-Loss and Brier Score).
   
    The calibration of the models was also evaluated with the calibration curve and adjusted with the Isotonic regulation.
   
    Finally using the SHAP Values ​​then the Features importance were evaluated.

    **4) Results**

    The evaluation of the models using the two selected metrics showed that XGBoost is the model that presents the lowest values ​​for them and therefore the one that produces the least error in evaluating the probability that an event belongs to one class or the other, 
    i.e. that a shot is a goal or not.

    ![Screenshot 2025-03-22 171351](https://github.com/user-attachments/assets/134f21e8-9103-4ef3-abac-8c164992be04)

    From the table you can see that both models manage to beat the performance of a random model, which I called Baseline, indicating that they are able to learn real patterns instead of making random guesses.

    And from both the metrics for the uncalibrated and calibrated models, it is clear that the XGBoost model is the best performing.

    The results of the PSxG models are shown below with the XGBoost SHAP Values ​​graphs.

    **Logistic Regression Results Visualization**
   
    ![Logistic_regression_classification](https://github.com/user-attachments/assets/9fd544bf-359a-4b65-b2ac-ed892007e6e8)

    **Random Forest Results Visualization**
   
    ![random_forest_classification](https://github.com/user-attachments/assets/7edbc345-e454-4e61-b524-28972e0e62ee)

    **XGBoost Results Visualization**
   
    ![xgboost_classification](https://github.com/user-attachments/assets/54bbf1a7-270d-4342-927c-4fbcd5c5cadd)

    **SHAP Values XGBoost**
   
    ![Figure 2025-03-02 205639 (27)](https://github.com/user-attachments/assets/21c8318a-f70d-45d2-9ce9-1e87128c7a4c)
   

    **SHAP Values Interpretation**
   
    From the SHAP values, it is clear that the final positions of the shot are some of the features that have the most impact on the model classification.

    From the plots of the shots on target, it is clear that most of the goals actually occur in the areas far from the center and closer to the posts.

    Finally, looking at the shots with a high probability of becoming goals, defined as those with PSxG>=0.90, it turns out that they are those with the lowest shot_distance, as shown by the SHAP graph, with few players in the shooting cone, often in shot_open_goal, 
    and as already said with the shot that ends as close as possible to one of the posts.

    **Shot_Speed problem**
   
    To close this article, a consideration on Shot_speed.

    From the SHAP values, it appears that as the speed of the shot increases, the probability of the shot converting into a goal decreases.

    This is counterintuitive and not very sensible, the reason must be sought in the calculation of this feature.

    Normally this feature is calculated with tracking data, while having only event data I had to calculate it as the ratio between the distance (approximated to a straight path) traveled by the ball divided by the duration of the shot event provided by statsbomb.

    In this case, in addition to the approximation made on the trajectory of the shot, a time interval calculated by an operator is used and a small error in the moment of selection of the start and end of the shooting event is enough to incur small errors on the 
    timestamps that have a high impact on the calculation of the speed.

    This probably leads to obtaining high speeds on long-distance shots due to errors in the calculation of the duration and distance of the shot and consequently associated with shots with a lower probability of being scored.

