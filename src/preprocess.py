import pandas as pd
import numpy as np
import os
from pymongo import MongoClient
from sklearn.ensemble import IsolationForest

from src.common.custom_logger import setup_logger

logger=setup_logger("preprocess")


class preprocess:
    def __init__(self,attributes=None,file_path=None):
        if attributes is None:
            attributes = {}
        self.uri = attributes.get('uri', None)
        self.database=attributes.get('database',None)
        self.collection=attributes.get('collection',None)
        self.IQR_collection=attributes.get('IQR_collection',None)
        self.ISO_collection=attributes.get('ISO_collection',None)
        self.file_path=file_path

    def load_mongo(self):
      # Replace with your MongoDB Atlas connection string
      client = MongoClient(self.uri)
      # Select your database
      db = client[self.database]
      collection = db[self.collection]

      # Fetch all documents in the collection
      documents = collection.find()
      data = list(documents)
      # Convert to DataFrame
      df = pd.DataFrame(data)
      # Exclude '_id' if needed
      df = df.drop(columns=['_id'])
      return df


    def load_dataset(self):
      if self.file_path is not None:
        df=pd.read_csv(self.file_path)
      else:
        df=self.load_mongo()

      return df


    # Function to remove outliers based on the IQR method
    def remove_outliers_ISO(self,df_main,label):
        df=df_main.groupby("label").get_group(label)
        logger.info(f"Number of eliment in {label} :{len(df)}")

        df=df.drop("label",axis=1)

        # 'contamination' defines the proportion of outliers you expect in the data
        iso_forest = IsolationForest(contamination=0.2, random_state=42,n_estimators=200,max_samples=512)
        df['outliers'] = iso_forest.fit_predict(df)
        # 'outlier' column contains -1 for outliers, 1 for normal data points

        # Show only the detected outliers

        # add 1 for any outlier in row and 0 for non outlier in row

        df_clean=df[df['outliers'] == 1] #filter

        logger.info(f"Number of outliers in {label} :{len(df[df['outliers'] == -1])}")
        logger.info(f"Number of Non outliers in {label} :{len(df_clean)}")

        df_clean=df_clean.drop("outliers",axis=1)

        #df_clean["label"]=label
        df_clean.loc[:, "label"] = label

        return df_clean

    def remove_outliers_IQR(self,df_main,label):
        df=df_main.groupby("label").get_group(label)
        logger.info(f"Number of eliment in {label} :{len(df)}")

        df=df.drop("label",axis=1)

        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        IQR = Q3 - Q1
        # Define outliers as values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

        # add 1 for any outlier in row and 0 for non outlier in row
        #df["outliers"]=[1 if x else 0 for x in outliers.any(axis=1)]
        df.loc[:, "outliers"] = [1 if x else 0 for x in outliers.any(axis=1)]

        df_clean=df[df['outliers'] == 0] #filter

        logger.info(f"Number of outliers in {label} :{len(df[df['outliers'] == 1])}")
        logger.info(f"Number of Non outliers in {label} :{len(df_clean)}")

        df_clean=df_clean.drop("outliers",axis=1)

        #df_clean["label"]=label
        df_clean.loc[:, "label"] = label

        return df_clean


    # Function to process each class
    def process_class(self,data_frame, class_name,value_column):
        logger.info(f"Processing class: {class_name}")
        df=data_frame.copy()
        # Filter by class
        class_df = df[df['label'] == class_name]

        # Show boxplot before removing outliers
        #plt.figure(figsize=(15, 10))
        #sns.boxplot(data=class_df[value_column])
        #plt.title(f'Boxplot Before Removing Outliers - Class {class_name}')
        #plt.show()

        # Remove outliers
        class_df_no_outliers_IQR = self.remove_outliers_IQR(df_main=df,label=class_name)
        class_df_no_outliers_ISO =self.remove_outliers_ISO(df_main=df,label=class_name)

        # Show boxplot after removing outliers
        #plt.figure(figsize=(15, 10))
        #sns.boxplot(data=class_df_no_outliers[value_column])
        #plt.title(f'Boxplot After Removing Outliers - Class {class_name}')
        #plt.show()

        return class_df_no_outliers_IQR,class_df_no_outliers_ISO

    def pre_process(self,df): 

        df["label"]=df["0"]
        df.drop("0",axis=1,inplace=True)
        logger.info(f"{df.info()}")
        df.head()
        logger.info(f"Null values {df.isnull().sum(axis=1)}")
        logger.info(f"Na values {df.isna().sum(axis=1)}")
        df.dropna(axis=0,inplace=True)
        df.drop_duplicates(inplace=True)
        logger.info(f"{df.describe(include='all')}")

        return df
    def load_dta_to_mongodb(self,df,collection_name):

        # Create a new client and connect to the server
        client = MongoClient(self.uri)

        # Send a ping to confirm a successful connection
        try:
            client.admin.command('ping')
            logger.info("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            logger.error(str(e))

        # Create (or connect to) the "asl_land_mark_detection" database
        db = client["asl_land_mark_detection"]

        # Create (or connect to) the "original" collection
        collection = db[collection_name]

        # Convert DataFrame to a list of dictionaries (MongoDB format)
        data = df.to_dict(orient="records")

        # Insert data into MongoDB
        collection.insert_many(data)

        logger.info(f"Data inserted successfully from {collection_name} Pandas DataFrame!")



    def main_exploratory(self,df):

        dataframe=self.pre_process(df)

        # List of classes to process
        classes = dataframe['label'].unique()
        Value=['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
          'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
          'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10',
          'y11', 'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20',
            'z0', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10','z11', 'z12', 'z13', 'z14', 'z15', 'z16', 'z17', 'z18', 'z19', 'z20'
          ]
        # Process each class and store the cleaned data
        cleaned_data_IQR = pd.DataFrame()
        cleaned_data_ISO = pd.DataFrame()

        for class_name in classes:
            class_df_no_outliers_IQR,class_df_no_outliers_ISO= self.process_class(dataframe, class_name,Value)  # Replace 'Value' with the actual column name
            cleaned_data_IQR = pd.concat([cleaned_data_IQR,class_df_no_outliers_IQR],axis=0,ignore_index=True)
            cleaned_data_ISO = pd.concat([cleaned_data_ISO,class_df_no_outliers_ISO],axis=0,ignore_index=True)

        logger.info(f'Cleaned IQR dataset :{cleaned_data_IQR.describe(include="all")}')
        logger.info(f'Cleaned ISO dataset :{cleaned_data_ISO.describe(include="all")}')
        # Save the cleaned dataset
        cleaned_data_IQR.to_csv(os.path.join("artifact/data",'ASL_cleaned_IQR_dataset2.csv'), index=False)
        cleaned_data_ISO.to_csv(os.path.join("artifact/data",'ASL_cleaned_ISO_dataset2.csv'), index=False)
        self.load_dta_to_mongodb(cleaned_data_IQR,self.IQR_collection)
        self.load_dta_to_mongodb(cleaned_data_ISO,self.ISO_collection)