import joblib
import pandas as pd
import numpy as np
import os
import random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

class ClusterPredict:
    def __init__(this):
        # load our model, load car data and preprocess it
        rootPath = os.path.abspath(os.path.join(__file__,"..\.."))
        modelFile = os.path.join(rootPath, "Cluster_Segmentation\\Car_Cluster44.pkl")
        this.model = joblib.load(modelFile)

        carDataFile = os.path.join(rootPath, "Cluster_Segmentation\\car_data.csv")
        car_df = pd.read_csv(carDataFile)
        car_df = car_df.rename(columns={"Market Category": "Market_Category", "Vehicle Size": "Vehicle_Size", "Vehicle Style": "Vehicle_Style" })
        car_df.drop(car_df.index[car_df['Market_Category'] == 'N/A'], inplace=True)
        car_df['Make_ENUM'] = car_df['Make']
        car_df['Year_ENUM'] = car_df['Year']
        car_df['Market_Category_ENUM'] = car_df['Market_Category']
        car_df['Vehicle_Size_ENUM'] = car_df['Vehicle_Size']
        car_df['Vehicle_Style_ENUM'] = car_df['Vehicle_Style']

        # add numeric converted columns to table
        M_List = car_df.Make.unique()
        MK_List = car_df.Market_Category.unique()
        VZ_List = car_df.Vehicle_Size.unique()
        VS_List = car_df.Vehicle_Style.unique()
        car_df.Make_ENUM = car_df['Make'].replace(M_List, range(len(M_List))) # for each unique category in column, replace with a number 0-(size of MList)
        car_df.Market_Category_ENUM = car_df['Market_Category'].replace(MK_List, range(len(MK_List)))
        car_df.Vehicle_Size_ENUM = car_df['Vehicle_Size'].replace(VZ_List, range(len(VZ_List)))
        car_df.Vehicle_Style_ENUM = car_df['Vehicle_Style'].replace(VS_List, range(len(VS_List)))

        this.car_df = car_df
    
    def predictCluster(this, data: list):
        # Predict cluster based on car data sample given
        data = np.array(data)
        result = this.model.fit_predict(data.reshape(1,-1))
        return result[0]

    def predictAllClusters(this):
        # Predict all clusters for each car in our dataframe
        predictors = this.car_df[['Make_ENUM', 'Year_ENUM', 'Market_Category_ENUM', 'Vehicle_Size_ENUM', 'Vehicle_Style_ENUM']].values
        predictors = preprocessing.StandardScaler().fit(predictors).transform(predictors.astype(float))
        result = this.model.fit_predict(predictors)
        return result

    def suggestThreeCars(this, cluster: int, printRes=False):
        # Find and recommend a list of 3 similar cars in recommended cluster
        result = this.predictAllClusters()
        res = np.where(result == cluster)
        suggestCars = [ str(this.car_df.iloc[i].Make)+" "+str(this.car_df.iloc[i].Model) for i in res[0] ]
        chooseThree = [ suggestCars[random.randint(0, len(suggestCars))] for i in range(0,3) ]
        if(printRes):
            print(f"\nHere are some cars you may like: {chooseThree[0]}, {chooseThree[1]}, {chooseThree[2]}")
        return chooseThree
