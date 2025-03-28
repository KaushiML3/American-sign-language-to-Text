from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot  as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from joblib import dump, load
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from pymongo import MongoClient
import os 


from src.common.custom_logger import setup_logger

logger=setup_logger("training")


class model_build():
  def __init__(self,attributes=None,file_path=None):
      if attributes is None:
          attributes = {}
      self.uri = attributes.get('uri', None)
      self.database=attributes.get('database',None)
      self.collection=attributes.get('collection',None)
      self.file_path=file_path
      self.le = LabelEncoder()
      self.lb = LabelBinarizer()
      self.X_train_lb=None
      self.X_test_lb=None
      self.y_train_lb=None
      self.y_test_lb=None
      self.X_train_le=None
      self.X_test_le=None 
      self.y_train_le=None
      self.y_test_le=None

      self.data_preprocessing()


  def load_mongo2(self):
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
      #df = df.drop(columns=['_id'])
      return df
  
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
  
  def data_loader(self):
      if self.file_path is not None:
        df=pd.read_csv(self.file_path)
        label=list(df["0"])
        df['Category']=[x.upper()for x in label]
        df.drop(['_id',],axis=1,inplace=True)
        df.head()
      else:
        df=self.load_mongo()
        logger.info(f"{df.head()}")
        label=list(df["label"])
        
        df['Category']=[x.upper()for x in label]
        #df.drop(['_id',],axis=1,inplace=True)
        df.head()

      return df

  def data_preprocessing(self):
    df=self.data_loader()
    
    # Fit and transform the 'Category' column
    df['Category_Encoded'] = self.le.fit_transform(df['Category'])

    # Fit and transform the 'Category' column
    df['Category_labelBinariz']= list(self.lb.fit_transform(df['Category']))
    Category_labelBinariz= list(self.lb.fit_transform(df['Category']))
    X=df.iloc[:,:63].values
    Y_le=df.iloc[:,-2].values
    #Y_lb=df.iloc[:,-1].values
    Y_lb=np.array(Category_labelBinariz)
    X=X/330
    logger.info(f"x shape : {X.shape}")
    logger.info(f"y shape : {Y_le.shape}")
    logger.info(f"y shape : {Y_lb.shape}")
    logger.info(df.head())
    self.X_train_le, self.X_test_le, self.y_train_le, self.y_test_le = train_test_split(X, Y_le, test_size=0.2, random_state=42,stratify=Y_le)
    self.X_train_lb, self.X_test_lb, self.y_train_lb, self.y_test_lb = train_test_split(X, Y_lb, test_size=0.2, random_state=42)
    logger.info(df.head())
    return df,self.X_train_le, self.X_test_le, self.y_train_le, self.y_test_le,self.X_train_lb, self.X_test_lb, self.y_train_lb, self.y_test_lb

  def plot_confusion_matrix(self,model, X_test, y_test,save_path):
    '''Function to plot confusion matrix for the passed model and the data'''

    sentiment_classes = self.le.classes_
    # use model to do the prediction
    y_pred = model.predict(X_test)
    # compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # plot confusion matrix
    plt.figure(figsize=(20,20))
    sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='d',
                xticklabels=sentiment_classes,
                yticklabels=sentiment_classes)
    plt.title('Confusion matrix', fontsize=16)
    plt.xlabel('Actual label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)



    # Save the plot
    plt.savefig(os.path.join(save_path,"confision_matrix"))
    plt.show()
    plt.close()  # Close the figure to prevent display issues



    logger.info('Classification Report:')
    report=classification_report(y_test, y_pred)
    logger.info(f"{classification_report(y_test, y_pred)}")


    # Define the file path
    report_path = os.path.join(save_path, "accuracy_report.txt")

    # Save accuracy and report to a TXT file
    with open(report_path, "w") as f:
        f.write("Classification Report:\n")
        f.write(report)

    logger.info(f"Accuracy report saved at: {report_path}")

    # Save the model to a file
    joblib.dump(model, os.path.join(save_path,'model.joblib'))

    # Save the LabelEncoder
    joblib.dump(self.le, os.path.join(save_path,'label_encoder.pkl'))


  def build_rf(self,save_path,n_estimators):
      rf = RandomForestClassifier(n_estimators=n_estimators,random_state=42)
      rf.fit(self.X_train_le, self.y_train_le)

      save_dir=os.path.join("artifact/model2",save_path)
      os.makedirs(save_dir, exist_ok=True)
      self.plot_confusion_matrix(rf, self.X_test_le, self.y_test_le,save_dir)

  def build_bg(self,save_path,n_estimators,max_depth):
      gbm = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=0.2, max_depth=max_depth, random_state=42,verbose= 1)
      gbm.fit(self.X_train_le, self.y_train_le)

      save_dir=os.path.join("artifact/model2",save_path)
      os.makedirs(save_dir, exist_ok=True)
      self.plot_confusion_matrix(gbm, self.X_test_le, self.y_test_le,save_dir)


  def build_CatBC(self,save_path,iterations,max_depth):
      catboost_model = CatBoostClassifier(iterations=iterations, learning_rate=0.2, depth=max_depth, verbose=1)
      catboost_model.fit(self.X_train_le, self.y_train_le)

      save_dir=os.path.join("artifact/model2",save_path)
      os.makedirs(save_dir, exist_ok=True)
      self.plot_confusion_matrix(catboost_model, self.X_test_le, self.y_test_le,save_dir)

  def build_xbc(self,save_path,params):
      dtrain = xgb.DMatrix(self.X_train_lb, label=self.y_train_lb)
      dtest = xgb.DMatrix(self.X_test_lb, label=self.y_test_lb )

      # Train the model
      bst = xgb.train(params, dtrain, num_boost_round=100)

      save_dir=os.path.join("artifact/model2",save_path)
      os.makedirs(save_dir, exist_ok=True)
      # Save the model to a file
      joblib.dump(bst , os.path.join(save_dir,'model.joblib'))
      # Save the LabelEncoder
      joblib.dump(self.lb, os.path.join(save_dir,'label_binanzer.pkl'))

      # Make predictions
      y_pred = bst.predict(dtest)
      y_pred_binary = (y_pred > 0.5).astype(int)
      # Evaluate the model
      accuracy = accuracy_score(self.y_test_lb , y_pred_binary)
      logger.info(f"Accuracy: {accuracy}")