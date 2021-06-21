import sys
from google.cloud import storage
import os
import json
import logging



class GSCBucketConnectorClass():

    def __init__(self):
        try:
            Path_to_credential = "D:/deeplearning learn/OrionEdgeSocialDistancingAPI/social_distance/config/las-prod-1-e312994d63d5.json"
            self.storage_client = storage.Client.from_service_account_json(Path_to_credential)  
        except:
            print(f"Failed to Initiate the GSCBucketConnectorClass ")


    def download_blob_4_detection(self,bucket_name,source_blob_Folder_name,destination_file_name):
        bucket=self.storage_client.get_bucket(bucket_name)
        blobs=list(bucket.list_blobs(prefix=source_blob_Folder_name))
        try:
            for blob in blobs:
                if(not blob.name.endswith("/")):
                    blob.download_to_filename(destination_file_name+blob.name)
        except:
            print(f"Failed to download the from bucket ")

bucket_name = "image_input"
destination_file_name = "D:/deeplearning learn/Social distancing/face-mask-detector/dataset_classifer_syngene/croppedface_14-12-2020/" 
#source_blob_Folder_name = "Syngene_S1_GF_entrance/MD/2020-12-14/"
#source_blob_Folder_name = "Syngene_S1_GF_Lift_Lobby/MD/2020-12-08/"
source_blob_Folder_name = "Syngene_S1_FF_Lift_Lobby/MD/2020-12-08/"
# Create this folder locally
if not os.path.exists(destination_file_name+source_blob_Folder_name):
    os.makedirs(destination_file_name+source_blob_Folder_name)

gcs=GSCBucketConnectorClass()
gcs.download_blob_4_detection(bucket_name,source_blob_Folder_name,destination_file_name)
            


        