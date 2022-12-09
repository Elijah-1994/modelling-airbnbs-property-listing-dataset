## Data Collection Pipeline Project
&nbsp;

The aim of this project is to utilise selenium webdriver and python methods to scrape text and image data from web html links of a chosen website and upload the python script and associated data onto dockerhub. The first step is to choose a website to scrape. It was descided to scrape text and image data of manga books on www.waterstones.com/ <br />
&nbsp;


## Milestone 1 - Prototype finding the individual page for each entry 
&nbsp;

__Setting up selenium__ 

The first step is to install chromedriver to google chrome. The chromedriver is sent to the relevant python path and selenium is installed using pip install selenium. Now the selenium module can be imported into the python script.
&nbsp;

![Selelium module](project_images/Milestone_1-Selenium-module.PNG)

*Figure 1 - selenium import in python*

&nbsp;

__Class WaterstonesScrapper.__

A class is coded which contains the various methods in order to scrape and store the required data. The def __init__ method was created in order to initialize the first instance of the class. in order to use selenium to connect to a website, the __webdriver.Chrome() method__ is stored in the self.driver variable this would allow selenium to connect to the google chrome browser. the __self.driver.get() method__  is used to allow selenium to drive towards waterstones.com 

&nbsp;

__accept_cookies() method:__

Once selenium drives towards the waterstones homepage there is an accept_cookies button which needs to be clicked on in order for the scrapping process to work. The __accept_cookies method__ consists of the code to complete this task. the first step is to inspect the html web elements on the waterstones website by pressing ctrl+c to find the element xpath file of the accept cookies button. 

&nbsp;

![Alt text](project_images/Milestone_1-accept_cookies_html.PNG)
*Figure 2 - html xpath of the accept cookies button*

 &nbsp;

The relative xpath was located and copy and pasted into the &ensp; __self.driver.find_element method__ &ensp; which allows the driver to point to the element. The &ensp; __accept_cookies_button.click() method__  &ensp;allows the webdriver to click on the accept cookies button on the waterstones website. The __time.sleep method__ is coded after so that the webdriver will wait a couple of seconds, so that the website doesn't suspect the user to be a bot.

![Alt text](project_images/Milestone_1-accept_cookies_method.PNG)
*Figure 3 - accept cookies button method*

&nbsp;

__navigate_to_manga_page_1 method__

This method is coded in order for the webdriver to navigate to the first page of the see more manga section. As with the __accept cookies method__ the first step is to inspect the html elements and find the relevant xpath in order to complete this task. 

&nbsp;

![Alt text](project_images/Milestone_1-inspect_manga_section.PNG)
*Figure 4 - manga section from page html elements*

&nbsp;

On inspection the html elements were contained within a tag which include a hyperlink reference 'href'. The html elements within the first page of the see more manga section were located within the html class='name' hence in order to store the hyperlinks the relative xpath are placed into the &ensp; __find_elements method__ &ensp; which returns the various web session links. in order to extract the html links a for loop was coded which iterates through each web element and calls the &ensp; __get.attribute('href') method__.  <br />
&nbsp;

Each link was then stored into a list. An if statement was coded in order to extract the correct html link from the web elements and is returned in the method as a string.

&nbsp;


![Alt text](project_images/Milestone_1-navigate_to_manga_page_1.PNG)
*Figure 5 - navigate_to_manga_page_1 method*

&nbsp;

__get_links_manga_page_1 method__
&nbsp;

The purpose of this method is to extract the html links of each manga books on page 1 and store them within a list. The html elements on page 1 were inspected to locate the html tags which store the href to each manga book on page 1. Once located the relative xpath are copied ito the __find_elements method__.

&nbsp;
![Alt text](project_images/Milestone_1-inspect_manga_section_page_1.PNG)
*Figure 6 - html elements on page 1*
&nbsp;

The method calls the __navigate_to_manga_page_1__ method which returns the html link of the see more manga section page 1 and the &ensp; __driver.get method__&ensp; is called so that the webdriver navigates to the first page. The __find_elements method__ is called to retrieve the web elements and then a for loop is coded in which the &ensp; __get.attribute('href') method__ &ensp; is called to extract the html link for each book on page 1 and  is appended to a list. The list along with the current url to page 1 is returned in a tuple.


&nbsp;
![Alt text](project_images/Milestone_1-get_links_manga_page_1.PNG)
*Figure 7 - get_links_manga_page_1 method*

&nbsp;

__get_links_manga_page_2_to_page_5 method__

&nbsp;

In order to expand the data extracted for this project it was decided to also scrape data from pages 2 to page 5 in the see more manga section. The purpose of this method is to store the html links of the books from page 2 to page 5 and append to the list of the html links extracted from page 1. The first step was to call the &ensp;__get_links_manga_page_1__&ensp; method which returns the url of see more mange section page .

&nbsp;

On inspection the url for pages 2 to the page 5 were similar to page 1 (minus the page number) therefore The string of the url was adjusted to 'https://www.waterstones.com/category/graphic-novels-manga/manga/page' and a for loop is was coded to update the url with the page numbers from 2 to 5 and these urls were saved in a list. The same methods to extract the html links were coded and the html links were appended to the list which contains the html links from page 1.

![Alt text](project_images/Milestone_1-get_links_manga_page_2_to_5.PNG)
*Figure 8 - get_links_manga_page_2_to_page_5 method*
&nbsp;

__scrapper  method__ 

This method contain the methods coded for milestone 1. This method is then called in a if __name__ == "__main__"  block.

&nbsp;
## Milestone 2 - Retrieve data from details page
&nbsp;

__create_directory method__

This method creates a folder directory to save the images scrapped from each book and the corresponding text data. This is done by importing os and applying the&ensp; __os.path.join method__.


![Alt text](project_images/Milstone-2%20-%20create%20directory.PNG)
*Figure 9 - create_directory method*

&nbsp;


__scrape_links_and_store_text_image_data method__

__Text data__
&nbsp;

This method is coded within a for loop which first scrapes the text data for each book and stores the data within a dictionary. As with the methods mentioned in milestone 1 the first step is to inspect the html elements to each link to find the xpath of the relevant data and place the xpaths into the &ensp;__find_elements method__&ensp;.  The text data included each books ISBN number, author, book format, and other information. Each dictionary is appended to a list. 

&nbsp;

![Alt text](project_images/Milestone_2%20-scrape_links_and_store_text_and_image_data.PNG)
*Figure 10 - scrapping and saving text data*


Each book is also assigned a unique id number (generated by importing the from uuid import uuid4 and calling the &ensp;__str(uuid4()) method__&ensp;, this id number is also be used to label each book image along with a timestamp (generated by importing the import time
from datetime import datetime and calling the &ensp;__datetime.now()__ &ensp; and &ensp;__time.strftime("%Y-%m-%d")__&ensp; methods.

__Image data__
&nbsp;

the method also finds the html element of each book element and calls the &ensp;__get_attribute('src') method__&ensp; to retrieve the src link for each image and then the &ensp;__requests.get().content method__&ensp; to retrieve the contents of each image(bytes). A context manger is coded in order to upload load each book image into the correct directory. This method returns the list which contains the dictionaries of the text data for each book.

&nbsp;

![Alt text](project_images/Milestone_2%20-scrape_links_and_store_text_and_image_data_2.PNG)
*Figure 11 - scrapping and saving image data*


&nbsp;

## Milestone 3 - Documentation and testing

&nbsp;

__Refactoring__

The first step was to review and refractor the code written in milestone 2. This included;

* Renaming methods and variables so that they are clear and concise to any who reads the script.
* Ensuring that the appropriate methods were made private.
* Re-ordering the sequence of the imports required for the code to run in alphabetical order.
* Adding docstrings to methods.

 These improvements makes the code look clearer and more user friendly.

&nbsp;

__Unit testing__

The second step was to set up unit tests for each public method. This was done by creating a test.py file which contains &ensp;__class producttestcase method__&ensp; to test each method. The main  purpose of tests is to ensure each public method returns the expected data type (string,list,dictionary) and to ensure the scrapper is correctly scrapping all the books from each page. This is to ensure that the code is processing the correct data as expected. Each unit test passed for each method.


__Project management__

The last step is to organise and add the relevant files which will ensure the scripts is packaged correctly. This includes adding;

* Renaming the python script as 'WaterstonesScrapper.py' and placing the script into a project folder.
* Placing the test file into a test folder.
* Creating a requirements.txt file which contains the external dependencies and versions.
* Creating a setup.py and setup.cfg which contains the meta data of the project and packages which need to be installed.
* Creating README.md file 
* Creating a license file which describes the license of the project.
* Creating a gitignore file.

## Milestone 4 - Containerising the scraper

&nbsp;

__Headless mode__

After confirming the unit tests still run, the next step was to run the scraper file in headless mode without the GUI. This was done so that the script could be run correctly in docker. The correct&ensp;__options arguments__&ensp; were coded into the __init method__ to allow the headless mode to work.

![Alt text](project_images/Milstone%204%20-%20options%20arguments.PNG)
*Figure 11 - Options arguments*


&nbsp;

__Docker image__

In order to build the docker image a docker file which contains the instructions on how to build the image is first created. A docker account was also created in order to upload the image file. The desktop app was downloaded.

The docker file contains the following;

* From - The base image for the docker image(python).
* Copy - Copies everything in the docker file directory (requirements.txt, scraper folder) into the container.
* Run -  Installs the required dependencies for the script to run. 
* CMD - Specifies the instruction that is to be executed when a Docker container starts.

&nbsp;

![Alt text](project_images/Milstone%204%20-%20docker%20file.PNG)
*Figure 12 - Dockerfile*


&nbsp;


The next step is to build the image using the docker build command.

&nbsp;

__Docker container__

&nbsp;

Now that the docker image is built the next step is to run the docker container using the docker run command. The script within the container ran fine with no issues. The container is then pushed onto docker hub.

&nbsp;

## Milestone 5 - Set up a CI/CD pipeline for your docker image

&nbsp;

in order to fully automate the docker image build and container run, it was first required to set up Github actions on the repository. 

__Create repository__
&nbsp;

The first step is to go yo the actions section in the repository on github and create two GitHub secrets actions. 

The first is a secret is called DOCKER_HUB_USERNAME which containes the name of the dockerhub account created and the second is called OCKER_HUB_ACCESS_TOKEN which contained a Personal Access Token (PAT) generated on dockerhub.

__Set up the workflow__
&nbsp;

The next step is to set up the GitHub Actions workflow for building and pushing the image to Docker Hub. This is done by going to the actions section on the repo and selecting set up workflow which creates a Github actions work file contained in yaml format.

&nbsp;

__Define the workflow steps__
&nbsp;

The  last step includes setting up the build context within the yaml file. The contains all the information for docker hub to copy to files mentioned in the dockerfile then build an image and automatically push to docker hub.

The last step is to commit the changes in the repo which would automatically start workflow. In order to make sure the workflow worked the image pushed on to docker hub was downloaded and a container was created and ran to ensure the script ran correctly.  A docker compose file which contains commands to self automate running containers was also created.