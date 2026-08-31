import hopsworks
from dotenv import load_dotenv
import os

load_dotenv()

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
print("Connected to project:", project.name)