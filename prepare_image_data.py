#%%
import boto3
import os 
from PIL import Image

def downloadDirectoryFroms3(bucketName, remoteDirectoryName):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucketName) 
    for obj in bucket.objects.filter(Prefix = remoteDirectoryName):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, obj.key) # save to same path
        
def resize_images(df_copy):
    print('hello')
        
filepath = "geeksforgeeks.png"
img = Image.open(filepath)
width = img.width
height = img.height
print("The height of the image is: ", height)
print("The width of the image is: ", width)



#downloadDirectoryFroms3('airbnb-property-listings','images/')