# -*- coding: utf-8 -*-

"""FUNCTIONS FOR FEATURES CALCULATION."""

import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder



"""1) FUNCTION TO EXTRACT COORDINATES."""
def extract_coordinates(row,column_name):

    #Extract coordinates from location columns
    #two dimension
    if len(row[column_name])==2:
        x_yard, y_yard = row[column_name]  
        z_meter=0
    
    #three dimension
    elif len(row[column_name])==3:
        x_yard, y_yard,z_yard = row[column_name] 
        
    return pd.Series([x_yard, y_yard,z_yard])



"""2) Shot Distance: From shooter to goal center"""
def Shot_distance_from_center(row):
    
    s_l=np.sqrt(np.square(120-row['location_x']) + np.square(40-row['location_y']))
    return pd.Series(s_l)
        
 
"""3) Shot Speed."""

def Shot_speed(row):
    
    if row['duration']!=0:
        
        s_l=np.sqrt(np.square(row['end_location_x']-row['location_x']) + np.square(row['end_location_y']-row['location_y']))

        speed=s_l/row['duration'] #Unit of mesure of Shot Speed is yard/s:

                                                   
        #Substitute unreal speed value (duration values in statsbomb data are irrealistic some times).
        #We substitute irrealistic values with the average shot speed (yard/second) value for different distance rane og shot recorded in professional football.
        if speed>=40:
            if 0<s_l<=10:
                speed=24.31
            elif 10<s_l<=20:
                speed=27.34
            elif s_l>20:
                speed=34.94
    else:
        speed=0
        
    return pd.Series(speed)

"""4) Shot Angle"""
def Shot_angle(row):
    
    x_palo_sinistro, y_palo_sinistro = 120, 36  # Right Post (Statsbomb values)
    x_palo_destro, y_palo_destro = 120, 44  # Left Post ( Statsbomb values)
    

    # Vectors A and B calculation
    A = np.array([x_palo_sinistro - row['location_x'], y_palo_sinistro - row['location_y']])
    B = np.array([x_palo_destro - row['location_x'], y_palo_destro - row['location_y']])    
    
    # Scalar product between A and B.
    dot_product = np.dot(A, B)
    
    # Leght of vectors (is the norm)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    
    #Cos angle calculation
    cos_theta = dot_product / (norm_A * norm_B)
    
    # Convert angle in radiant
    theta_radians = np.arccos(cos_theta)
    
    # Convert angle in degree
    theta_degrees = np.degrees(theta_radians)
    
    return pd.Series(theta_degrees)



"""Functions to calculate if a player is in the shooting corner"""
# Function to calculate the angolo between two vectors
def angolo_tra_vettori(v1, v2):
    prodotto_scalare = np.dot(v1, v2)
    magnitudine_v1 = np.linalg.norm(v1)
    magnitudine_v2 = np.linalg.norm(v2)
    cos_theta = prodotto_scalare / (magnitudine_v1 * magnitudine_v2)
    # Arcsine to find the angle in radians, then we convert it to degrees
    angolo = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angolo)

# Function to calculate the directional vector between two points
def vettore(punto_a, punto_b):
    return (punto_b[0] - punto_a[0], punto_b[1] - punto_a[1])




"""Function to calculate the number of players (including teammates and goalkeeper) in the shooting cone."""
def number_of_defenders_inside_shot_cone(row):
    
    #Extract coordinates of shot
    x_T,y_T=row['location'][0], row['location'][1]
    
    #Extraction of freeze frame
    freeze_frame = row['shot_freeze_frame']
    
    #Define variables to save number of players found inside shot cone and cone density
    number_of_defenders_inside=0
    cone_density=0
    if isinstance(freeze_frame, list):

        #Start iteration.
        for i in range(len(freeze_frame)):
            
            player_pos = freeze_frame[i]['location']
            x_yard, y_yard = player_pos[0],player_pos[1]  # player positions extraction.

            # Define Shooter, goalpost and defender positions

            tiratore = (x_T,y_T)  # Coordinates (x, y) of the shooter
            palo1 = (120, 36)       # Coordinates of the first post statsbomb in meters
            palo2 = (120, 44 )    # Coordinates of the second post statsbomb in meters
            difensore = (x_yard, y_yard) # Player coordinate
        
            # Vectors between shooter and goal posts
            vettore_tiratore_palo1 = vettore(tiratore, palo1)
            vettore_tiratore_palo2 = vettore(tiratore, palo2)
            
            # Vector between shooter and defender
            vettore_tiratore_difensore = vettore(tiratore, difensore)
        
            #Calculating the angle between the shooter and the two goal posts
            angolo_tiratore_pali = angolo_tra_vettori(vettore_tiratore_palo1, vettore_tiratore_palo2)
        
            # Calculating the angle between shooter -> post1 and shooter -> defender
            angolo_difensore_palo1 = angolo_tra_vettori(vettore_tiratore_palo1, vettore_tiratore_difensore)
        
            # Calculating the angle between shooter -> post2 and shooter -> defender
            angolo_difensore_palo2 = angolo_tra_vettori(vettore_tiratore_palo2, vettore_tiratore_difensore)
            
            # The defender is in the cone if the angle to both posts is less than the angle between the posts.
            if angolo_difensore_palo1 <= angolo_tiratore_pali and angolo_difensore_palo2 <= angolo_tiratore_pali:
                
                #Add to the variable number of players inside cone
                number_of_defenders_inside+=1
                
                #inverse of the distance calculation.
                #Distance
                d=np.sqrt(np.square(x_yard-x_T) + np.square(y_yard-y_T))
                
                #Add to the variable cone density
                cone_density+=(1/d)
                cone_density=round(cone_density,2)
    
    #If these aren't players in the freeze frame
    else:
         number_of_defenders_inside=None
         cone_density=None
         
    return number_of_defenders_inside,cone_density




""""Function for CALCULATE DISTANCE TO THE CLOSEST SHOOTERS"""
def Distance_to_D1_and_D2(row):
    #Extract coordinates of shot
    x_T,y_T=row['location'][0], row['location'][1]

    # Freeze frame Extraction
    freeze_frame = row['shot_freeze_frame']
    
    distance_list=[]
    #Check if the freeze_frame is present (It is not in the penalties)
    if isinstance(freeze_frame, list):

        #Start to calculate distances
        for i in range(len(freeze_frame)):
            
            #Check if is a opponent
            teammate = freeze_frame[i]['teammate']
            if teammate==False:
                player_pos = freeze_frame[i]['location']
                x_yard, y_yard = player_pos[0],player_pos[1]  # Defender position extraction.
                #Distance calculation.
                d=np.sqrt(np.square(x_yard-x_T) + np.square(y_yard-y_T))
                #insert values inside list
                distance_list.append(d)
        
        #Sort the list so that I have the two lowest values ​​in position 0 and 1.
        distance_list_sorted  =sorted(distance_list, reverse=False)  # Sort list
        if len(distance_list_sorted)>=2:
            distance_to_d1=distance_list_sorted[0]
            distance_to_d2=distance_list_sorted[1]
        else:
            distance_to_d1=distance_list_sorted[0]
            distance_to_d2=None

    #If these aren't players in the freeze frame            
    else:
         distance_to_d1=None
         distance_to_d2=None
                
    return  distance_to_d1,distance_to_d2



"""Function to Calculate the distance between the goalkeeper and the shooter."""

def Distace_to_keeper_and_coordinate(row):
    #Extract coordinates of shot
    x_T,y_T=row['location'][0], row['location'][1]
    # Freeze frame Extraction
    freeze_frame =row['shot_freeze_frame']
    distance_to_keeper=0
    x_yard=0 
    y_yard=0
    #Check if the freeze_frame is present (It is not in the penalties)
    if isinstance(freeze_frame, list):

        for i in range(len(freeze_frame)):
            #Extract the position
            position_dict = freeze_frame[i]['position']
            position=position_dict['name']
            
            #Check if is a opponent
            teammate = freeze_frame[i]['teammate']
            
            if position=='Goalkeeper':
                #If is it a opponents of Keeper proceed to calculate the distance
                if teammate==False:
                    #Distance calculation.
                    player_pos = freeze_frame[i]['location']
                    x_yard, y_yard = player_pos[0],player_pos[1]  # Estrai x e y dalla lista
                    distance_to_keeper=np.sqrt(np.square(x_yard-x_T) + np.square(y_yard-y_T))#Distance calculation
                    break
                
                #In some freeze frames there are errors and the goalkeeper is considered a teammate. 
                #it is logical to think that if there is only one goalkeeper it is usually the opposing one and sometimes it is reported as a teammate 
                #I checked via video I discovered this problem.
                #In these case if there is only one goalkeeper you treat him as an opponent                                  
                elif teammate==True:
                    player_pos = freeze_frame[i]['location']
                    x_yard, y_yard = player_pos[0],player_pos[1]  # Estrai x e y dalla lista
                    distance_to_keeper=np.sqrt(np.square(x_yard-x_T) + np.square(y_yard-y_T))#Distance calculation
        
        #In case there isn't GoalKeeper information inside freeze frame
        if distance_to_keeper==0 and x_yard==0 and y_yard==0:
            distance_to_keeper=None
            x_yard=None
            y_yard=None
            
        
                        
    #If these isn't freeze frame            
    else:
         distance_to_keeper=None
         x_yard=None
         y_yard=None
         
    return distance_to_keeper,x_yard,y_yard


"""Function to calculate distance between GK and Goal center"""
def Distace_to_keeper_and_goal_center(row):
    # Freeze frame Extraction
    freeze_frame =row['shot_freeze_frame']
    distance_keeper_to_goal_center=0
    x_yard=0 
    y_yard=0
    #Check if the freeze_frame is present (It is not in the penalties)
    if isinstance(freeze_frame, list):

        for i in range(len(freeze_frame)):
            #Find the position
            position_dict = freeze_frame[i]['position']
            position=position_dict['name']
            #Teammate info extraction
            teammate = freeze_frame[i]['teammate']            
            if position=='Goalkeeper':
                #Check if is a opponent the GoalKeeper
                if teammate==False:
                    #distance calculation.
                    player_pos = freeze_frame[i]['location']
                    x_yard, y_yard = player_pos[0],player_pos[1]  # Player position extraction.
                    distance_keeper_to_goal_center=np.sqrt(np.square(x_yard-120) + np.square(y_yard-40))
                    distance_keeper_to_goal_center=round(distance_keeper_to_goal_center,2)
                    break
                
                #In some freeze frames there are errors and the goalkeeper is considered a teammate. 
                #it is logical to think that if there is only one goalkeeper it is usually the opposing one and sometimes it is reported as a teammate 
                #I checked via video I discovered this problem.
                #In these case if there is only one goalkeeper you treat him as an opponent              
                elif teammate==True:
                    player_pos = freeze_frame[i]['location']
                    x_yard, y_yard = player_pos[0],player_pos[1]   # Player position extraction.
                    distance_keeper_to_goal_center=np.sqrt(np.square(x_yard-120) + np.square(y_yard-40))
                    distance_keeper_to_goal_center=round(distance_keeper_to_goal_center,2)

        #In case there isn't GoalKeeper information inside freeze frame
        if distance_keeper_to_goal_center==0 and x_yard==0 and y_yard==0:
            distance_keeper_to_goal_center=None
            
        
                        
    #In case there isn't freeze frame
    else:
         distance_keeper_to_goal_center=None

         
    return distance_keeper_to_goal_center

"""Function to calculate Keeper Angle"""
def Angle_Keeper_and_Posts(row):
    #Coordinate Pali Yard
    x_palo_sinistro, y_palo_sinistro = 120, 36  # Right Post (Statsbomb values)
    x_palo_destro, y_palo_destro = 120, 44  # Left Post (Statsbomb values)
    
    
    # Freeze frame Extraction
    freeze_frame =row['shot_freeze_frame']
    theta_degrees_keeper=0
    x_yard=0 
    y_yard=0
    
    #Check if the freeze_frame is present (It is not in the penalties)
    if isinstance(freeze_frame, list):

        for i in range(len(freeze_frame)):
            #Extract the position
            position_dict = freeze_frame[i]['position']
            position=position_dict['name']
            #Teammate info extraction
            teammate = freeze_frame[i]['teammate']
            if position=='Goalkeeper':
                #Check if is a opponent the GoalKeeper
                if teammate==False:
                    #Calcolo la distanza.
                    player_pos = freeze_frame[i]['location']
                    x_yard, y_yard = player_pos[0],player_pos[1]  # Player position extraction.
                    #Put the condition that if the coordinates are equal to one of the 2 poles the angle is zero.
                    #if not by default it will set it to NAN
                    if x_yard==x_palo_destro and (y_yard==y_palo_sinistro or y_yard==y_palo_destro):
                        theta_degrees_keeper=0
                        break
                    else:
                        #Calculate vectors A and B
                        A = np.array([x_palo_sinistro - x_yard, y_palo_sinistro - y_yard])
                        B = np.array([x_palo_destro - x_yard, y_palo_destro - y_yard])    
                        
                        #Calculate the scalar product between A and B
                        dot_product = np.dot(A, B)
                        
                        # Calculate the lengths of the vectors, which is then the norm
                        norm_A = np.linalg.norm(A)
                        norm_B = np.linalg.norm(B)
                        
                        # Calculate the cosine of the angle
                        cos_theta = dot_product / (norm_A * norm_B)
                        
                        # Calculate the angle in radians
                        theta_radians = np.arccos(cos_theta)
                        
                        # Convert angle to degrees
                        theta_degrees_keeper = np.degrees(theta_radians)   
                        theta_degrees_keeper=round(theta_degrees_keeper,2)

                        break
                         
                #In some freeze frames there are errors and the goalkeeper is considered a teammate. 
                #it is logical to think that if there is only one goalkeeper it is usually the opposing one and sometimes it is reported as a teammate 
                #I checked via video I discovered this problem.
                #In these case if there is only one goalkeeper you treat him as an opponent and reduce the distance      
                elif teammate==True:
                    player_pos = freeze_frame[i]['location']
                    x_yard, y_yard = player_pos[0],player_pos[1]  # Estrai x e y dalla lista
                    #Put the condition that if the coordinates are equal to one of the 2 poles the angle is zero.
                    #if not by default it will set it to NAN
                    if x_yard==x_palo_destro and (y_yard==y_palo_sinistro or y_yard==y_palo_destro):
                        theta_degrees_keeper=0
                        break
                    else:
                        #Calculate vectors A and B
                        A = np.array([x_palo_sinistro - x_yard, y_palo_sinistro - y_yard])
                        B = np.array([x_palo_destro - x_yard, y_palo_destro - y_yard])    
                        
                        #Calculate the scalar product between A and B
                        dot_product = np.dot(A, B)
                        
                        # Calculate the lengths of the vectors, which is then the norm
                        norm_A = np.linalg.norm(A)
                        norm_B = np.linalg.norm(B)
                        
                        # Calculate the cosine of the angle
                        cos_theta = dot_product / (norm_A * norm_B)
                        
                        # Calculate the angle in radians
                        theta_radians = np.arccos(cos_theta)
                        
                        # Convert angle to degrees
                        theta_degrees_keeper = np.degrees(theta_radians)   
                        theta_degrees_keeper=round(theta_degrees_keeper,2)
                        break
                    
        #Se non ha trovato niente e i valori inizali sono uguali a zero vuol dire che on c'è dato sul portiere per calcolare
        if theta_degrees_keeper==0 and x_yard==0 and y_yard==0:
            theta_degrees_keeper=None
            
    #In case there isn't GoalKeeper information inside freeze frame
    else:
         theta_degrees_keeper=None
    
    return pd.Series(theta_degrees_keeper)






