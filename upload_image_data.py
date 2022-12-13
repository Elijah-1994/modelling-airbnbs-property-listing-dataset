#%%
import boto3 
import s3fs
s3_file = s3fs.S3FileSystem()
local_path = "airbnb-property-listings/images"
s3_path = "airbnb-property-listings/airbnb-property-listings/images"
s3_file.put(local_path, s3_path, recursive=True) 
#%%

