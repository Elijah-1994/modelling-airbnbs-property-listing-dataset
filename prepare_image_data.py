from PIL import Image 
from time import sleep
import boto3 
import os.path

def downloadDirectoryFroms3(bucketName, remoteDirectoryName):
    '''
        This function downloads the images from the 'airbnb-property-listings' bucket and saves into to the images folder

    '''   
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucketName) 
    for obj in bucket.objects.filter(Prefix=remoteDirectoryName):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, obj.key)

def create_directory() -> str:
    '''
        This function creates the working directory to save the processed airbnb images.

        Returns:
            string: returns the path of the processed images directory.
            
    '''
    processed_images_directory = 'processed_images' 
    parent_directory = os.getcwd()
    path = os.path.join(parent_directory, processed_images_directory)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print(f'{error} has occurred while creating directory.')
    return path

def calculate_smallest_image_height(path) -> int:
    '''
        This function calculates the smallest image height of the airbnb property images.

        Returns:
            integer: returns the smallest image height.
            
    '''
    image_height = []
    for dirpath, _, filenames in os.walk(path):
        for path_image in filenames:
            image = os.path.join(dirpath, path_image)
            with Image.open(image) as img:
                height = img.height
                image_height.append(height)
    minimum_height = min(image_height)
    return minimum_height

def delete_image_by_mode(mode, file_path, img):
    '''
        This function determines the image mode and deletes image directory if image mode in not RGB.
   
    '''
    if img.mode != mode:
            sleep(5)
            os.remove(file_path, dir_fd=None)

def resize_images(path):
    '''
        This function resizes the airbnb images and saves them in a folder.
   
    '''
    for dirpath, _, filenames in os.walk(path):
        for path_image in filenames:
            base_height = calculate_smallest_image_height(path)
            image = os.path.join(dirpath, path_image)
            with Image.open(image) as img:
                width = img.width
                height = img.height
                aspect_ratio = width / height
                new_width = int(base_height*aspect_ratio)
                img = img.resize((base_height,new_width), Image.Resampling.LANCZOS)
                image_path_and_name = os.path.split(image) 
                image_name_and_ext = os.path.splitext(image_path_and_name[1]) 
                name = image_name_and_ext[0] + '.png'
                file_path = os.path.join(create_directory(), name)
                img.save(file_path)
                delete_image_by_mode('RGB', file_path, img)

if __name__ == '__main__':
    #downloadDirectoryFroms3('airbnb-property-listings', 'images')
    resize_images('images')
             

