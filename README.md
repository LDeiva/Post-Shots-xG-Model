# Post-Shots-xG-Model
Creation a little Post-Shots xG model using Free Statsbomb Event Data.

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

3) **Creation of target variable**:
   
   The Targhet variable was created converting the description of the final result of the shot into **0** if it was a shot blocked by the goalkeeper and into **1** if it was converted into a Goal.
 
4) **Model Selection**:
   
   Since we are dealing with a binary classification problem, the models that are best suited to this type of problem are the following classifiers:
   
   • **Logistic Regression**:
      
     Logistic regression is a linear classification model.
   
     It is based on the sigmoid function which predicts the probability that a given example belongs to a given class in our case 1 (Goal) or 0 (No Goal).
   
     ![image](https://github.com/user-attachments/assets/956d0846-ef6d-4d8d-8995-c354e7de7e5c)

     • P(Goal) = Probability of the shot becoming a goal.
   
     • e = Euler’s number (~2.718).
   
     • β₀ = The intercept (a baseline value).
   
     • βᵢ = Weights assigned to different shot factors.
   
     • xᵢ = The different shot-related variables (e.g., distance, angle).

   • **Random Forest**:

     ![image](https://github.com/user-attachments/assets/0a003b96-ac4b-4f7b-a21a-bae53ae788cf)

      The random forest is an ensemble model, that is, it combines different models (in this case, multiple decision trees), thus increasing accuracy and decreasing the risk of overfitting.

      Each tree is trained on a subset of the dataset to decrease variance and only on a subset of features, so as to prevent some trees from becoming dominant.
   
      Finally, the class prediction is obtained with the majority vote of the trees.

   • **XGBoost**:
   ![image](https://github.com/user-attachments/assets/4b355137-05e4-460f-8c4e-25dac313a058)

      Unlike the random forest that creates independent trees (Bagging) it creates trees in sequence (Boosting).
   
      In practice each new tree corrects the errors made by the previous one.
   
      The trees are trained to minimize the residual error using the gradient of the loss function.
   
      XGBoost adds L1 and L2 regularizations to minimize the possibility of overfitting and can also automatically handle missing values ​​in the dataset.

5) **Metric Selection**

     **A) Log-Loss**
   
      ![image](https://github.com/user-attachments/assets/d5ca30f3-940c-49e9-bb8a-0b41ff66c3f7)
   
      • y<sub>i</sub> = is the actual result (1 = goal, 0 = no goal).
   
      • p<sub>i</sub> =is the probability predicted by the model.
   
      • N = Number of shots.

      Unlike other metrics, Log-Loss evaluates how close the predicted probabilities are to the actual outcomes (goal or no goal).
      

      

      






