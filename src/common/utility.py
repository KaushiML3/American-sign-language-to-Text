
import yaml
import os
import csv
from datetime import datetime
from roboflow import Roboflow


#from common.custom_logger import setup_logger
from src import setup_logger


logger=setup_logger("utility")

def read_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            logger.info(f"Read the yaml file in {file_path}")
            return data
    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")
        return None
    

def write_yaml(save_path, data):
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Write YAML data to file
        with open(save_path, "w") as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)

        logger.info(f"Write the yaml file in {save_path}")

    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")
        return None



def download_roboflow_dataset(api_key, workspace, project_name, version_number, dataset_format):
  try:
    # Initialize Roboflow client
    rf = Roboflow(api_key=api_key)
    
    # Ensure `project_name` is passed as a string
    project = rf.workspace(workspace).project(project_name)  
    
    # Ensure `version_number` is an integer
    version = project.version(version_number)  
    
    # Use `project_name` instead of `project` (which is an object)
    save_dir = os.path.join("artifact","dataset",project_name, f"v{version_number}")
    
    # Create the directory if it doesn't exist
    #os.makedirs(save_dir, exist_ok=True)
    
    # Download dataset to the specified directory
    dataset = version.download(dataset_format, location=save_dir)
    logger.info(f"Save the {project_name} dataset  in {save_dir}")
    return save_dir

  except Exception as e:
    logger.info(f"An error occurred: {str(e)}")
    return None



# Function to log model details in CSV
def log_model_details(name, version, model_path):
    try:
        csv_path="artifact/models.csv"
        file_exists = os.path.exists(csv_path)

        # Open CSV file in append mode
        with open(csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)

            # Write header if file is newly created
            if not file_exists:
                writer.writerow(["name", "version", "model_path", "timestamp"])

            # Write model details
            writer.writerow([name, version, model_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        logger.info(f"Model details logged in: {csv_path}")

    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")
 


# Function to log model details in CSV
def log_dataset_details(name, version, model_path):
    try:
        csv_path="artifact/dataset.csv"
        file_exists = os.path.exists(csv_path)

        # Open CSV file in append mode
        with open(csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)

            # Write header if file is newly created
            if not file_exists:
                writer.writerow(["name", "version", "datsset_path", "timestamp"])

            # Write model details
            writer.writerow([name, version, model_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        logger.info(f"Dataset details logged in: {csv_path}")

    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")




# Function to get the latest model version and path for a specific model name
def get_latest_model_version(model_name):
    try:
        csv_path=os.path.join("artifact","models.csv")
        if not os.path.exists(csv_path):
            logger.info("No models logged yet.")
            return None, None

        with open(csv_path, mode="r") as file:
            reader = csv.DictReader(file)
            models = [row for row in reader if row["name"] == model_name]  # Filter by model name

        if not models:
            logger.info(f"No models found for '{model_name}'.")
            return None, None

        # Sort models by version number (assuming versions are in format "v1", "v2", ...)
        models.sort(key=lambda x: int(x["version"].lstrip("v")))  

        # Get the latest model entry
        latest_model = models[-1]
        return int(latest_model["version"]), str(latest_model["model_path"])
    
    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")
        return None,None

    