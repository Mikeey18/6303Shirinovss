import os

class Config:
    PAINTINGS_DIR = os.path.join(os.getcwd(), 'paintings')
    CSV_PATH = './data/MetObjects.csv'
    MET_API_URL = "https://collectionapi.metmuseum.org/public/collection/v1"