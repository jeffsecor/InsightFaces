# Predicting Project Themes Based on Facial Features of Insight Data Science Fellows

The following will be a journal style entry as I create the program. There will also be another file with the self-contained python code.  The question we are after is "can we predict the type of project a fellow will choose based on their facial features?" My intial guess is 'no' because I suspect that the common characteristic of Insight fellows is their knowledge and technical methods, not their appearance.  But, it seems like a fun question to ask, and also presents a very in depth learning experience for a novice programmer such as myself.  Im learning as I go, so this may get a little bumpy.

The general plan is:
1. Scrape image files and project text from the webpage
2. Identify classes/groups/types of projects
3. Learn facial recognition packages and identify facial characteristics
4. Look at correlations of selected facial features to projects

## 1. Scraping Images and Corresponding Project Text
As one would expect from a data science fellowship program, the Insight website is well written and the page source is dense and complex, but easy(ish) to navigate.  To start, we will pull the webpage using beautiful soup

```python
import bs4  # BeautifulSoup4 to scrape the webpage#
url = 'https://www.insightdatascience.com/fellows'
response=requests.get(url)
html=response.content
soup=bs4.BeautifulSoup(html,'lxml')
```

Looking at the page source (left click and 'view source') shows us how to find the images.  Each fellows image exists inside a div class 'fellow_img' (Im new to html, so apologies for any incorrect terminology).  So let's make a list containing all the image urls.  We use beautiful soup to find all the 'fellow_img' class data
```python
data=[info for info in soup.find_all(class_= "fellow_img")]
```
The size of our list can be found with **len(data)** which tells us that there are 794 entries.  Lets take a look at one of them with **print(data[5])** which outputs
```python
<div class="fellow_img" style='background-image:url("https://daks2k3a4ib2z.cloudfront.net/575a31d2ce5d01dc7a20de45/5769bb9322c61e320363317d_Jessica-Pic.jpg")'><div class="fellow_caption_back"><div class="fellow_name">Jessica Zúñiga</div><div class="fellow_company">LinkedIn</div></div></div> 
```
The url is in the 'background image' style.  It took a bit for me to figure this out.  One thing that helped is to check the type of this thing by **type(data[5])** which tells us that it is a 'bs4.element.Tag'.  That means we can use some function of beautiful soup to extract the url information.  A bit of searching leads to the right sytax, which is pleasantly intuitive.  The text of the style section is pulled with **data[5]["style"]** which outputs 'background-image:url("https://daks2k3a4ib2z.cloudfront.net/575a31d2ce5d01dc7a20de45/5769b885e8628b5103c73621_davidfreeman-main.jpg")'. (n.b. I later realized that **['style']** can be added on to the above list comprehension to eliminate the intermediate step of scraping the text) Checking **type(data[5]['style'])** tells us that it is a string , which means we can use regex to pull out the url.  Since they are all http....jpg, the regex is straightforward.  Except the group function is needed to output the match as a string instead of a regex match result.  

```python
import re
match=re.search(r'https.*jpg',text).group()
```
which outputs the url for the image.  Alltogether the code up to this point is:
```python
import re # pull url of image name from soup
import requests #to pull html pages#
import bs4 #BeautifulSoup4 to scrape the webpage#



url = 'https://www.insightdatascience.com/fellows'
response=requests.get(url)
html=response.content
soup=bs4.BeautifulSoup(html,'lxml')


data=[info for info in soup.find_all(class_= "fellow_img")]
print(len(data))


text=data[5]['style']
match=re.search(r'https.*jpg',text).group()
print(match)
```
This works to pull out the url from the first element in the list **data**.  The next thing we want is to catch all the urls from the soup.  I found that the **.group()** was giving an error becuase the **.group()** function could not act on a type "none".  Checking the documentation of the regular expression functions, 'none' is returned when there is not a match, and so there is no string to group together.  To find the root of the problem, I printed each match as it came out in the loop, along with the text it was trying to match, and found that some images are png, while others are JPG, PNG, or in some cases there are no images.  To work around this,  we can change our matching conditions to be case insensitive using **(i?)** and also an 'or' condition.  The matching expression now looks like 
```python
match=re.search(r'https.*jpg(?i)|png(?i)',text).group()
```
The missing images will also throw an error from the **.group()** function, so a **try...except** structure is appropriate.  We make an empty list for the image urls and run a for loop.  The print statements at the end are to check to fidelity of the data and regex
```python
url_list=[]
for i in range(len(data)):
    try:
        url_list.append(re.search(r'https.*jpg(?i)|https.*png(?i)',data[i]['style']).group())
    except:
        url_list.append('No Image')
print(len(data))  #check length of data list
print(len(url_list))  #check this list is the same length
print(data[402],"\n\n",url_list[402],"\n") #compare random entries of each list
print(data[-1],"\n\n",url_list[-1],"\n")
```
There we have it, a list of image urls ready to be called on to download the images.  

The next step in the data collection and cleaning is to get the name of the projects that correspond to the headshots. Similar to how we found the image urls, we look at the webpage source and see that the project title is listed under a class 'tootip_project'.  This scraping step is a bit simpler beacause the information is the only text in the tags, so a single list comprehension with the **.text** function will work.  And remember to check that the project list has the same length as the url_list.  Finally, double check a few values against the webpage to make sure the projects correspond to the right peoples faces
```python
projects=[project.text for project in soup.find_all(class_= "tooltip_project")]
```
And thats it for the project list, pretty simple there.

To finish this section, lets combine the url with the project title so that its ready for the image.  A dictionary is not too helpful, becuase we dont want the url as a key since it is too obscure.  In this case. a pandas data frame works great. The topic column will be used when we classify each project into a general topic  A few lines to set this up
```python
import pandas as pd
x=pd.DataFrame(columns=('url','project',topic))
x.url=url_list
x.project=projects
```
## 2. Identify Types of Projects

We'll need to make a finite set of project categories.  To do so, we'll use a bag of words approach in order to find the most common themes.  Our dataframe column 'project' contains all the cleaned project information.  (The list 'projects could be used if it were cleaned of linespaces and other special characters, but this was done by the dataframe).  To make a bag or words, use a Counter from the collections package.  In the counter, we use the regular expression to pick out any word **longer than 3 characters** and make each word lowercase to avoid duplication.  The way that this counter works is for each item in **list(x.project)**, so we sum all the bags together in order to found the counts for all the projects.  Then a bar plot will show us the most common words.  We see that there are still a lot of generic words, so we want to remove these from our data.  We can do this with an ignore list, and then search the counter dictionary for these ignore words.  If we run the program a few times we can see what new generic words show up and we add them to the ignore list.  Finally, we pick out the most common words and plot them.  Here is what there is so far:
```python
# make a list fellow's projects
projects=[project.text for project in soup.find_all(class_= "tooltip_project")] 

# create data frame 
x=pd.DataFrame(columns=('url','project','topic'))
x.url=url_list
x.project=projects


#find words in projects that are longer than 3 characters and make all lowercase
#to avaoid doubles from capitalization
words = list(x.project)
bagsofwords = [collections.Counter(re.findall(r'\w{4,}', word.lower())) for word in words] 
sumbags=sum(bagsofwords,collections.Counter())

#delete common words
ignore = {'your','find','with', 'from','where','that','will','what','they','based',
          'next','using','recommender','predicting','predict','recommend'}
for word in ignore:
	if word in sumbags:
		del sumbags[word]

#find most common value data set from couter object
number = 50
common_words=[wordval[0] for wordval in sumbags.most_common(number)]
common_values=[wordval[1] for wordval in sumbags.most_common(number)]

#plot histogram of common words
plt.bar(common_words,common_values,color='g')
plt.xticks(rotation=90)
plt.tight_layout(pad=2)
plt.title('%d Most Common Words in Insight Data Fellows Project Titles'%number)
plt.ylabel('Counts')
plt.show()
```
![Common Words](https://github.com/jeffsecor/InsightFaces/blob/master/wordChart2.png)

This is good, we see a few topics in here. Also note that the most common word is 'predicting' which is the first word of this project!!!  If we look at the full project list, we see that there are some combination words, like 'FacebookDigest' that are not properly counted by this method,  but we will do our best for now.  Also, there are still some duplicates, like 'rccomend' and 'recomender', or 'predict' and 'predictor' that we want to combine.  We want to group each together and then add up the occurances for the composite group. So let's try to write a few lines that can do this.

This turns out to be very difficult to categorize the projcects.  One attempt to find duplicates is the following:
```python
for key in sumbags:
	for key2 in sumbags:
		if key in key2 and key!=key2:
			print(key,key2)
```
This takes each key, then scans it across all the keys in the dictionary.  If the string is a strict subset, i.e. is part of another string but not the same so that it does not match itself, then it prints out the pair.  In total, there are 1093 pairs.  This is not a good filter because it is matching things like 'test' and 'greatest'.  Another option is to scan if the key string is the start of another key like the predictor example above
```python
for key in sumbags:
	for key2 in sumbags:
		try:
			re.search(r'^%s.+'%key2,key).group()
			j+=1
			print(key,key2)
		except:
			pass
```
This is a bit closer to what we need, but also does not work as can be seen in the example 'fair' and 'fairbnb', or 'come' and 'comedy' or 'care' and 'career'

????????????????????????????

Ok, lets roll up our sleeves and get it done.  A bit of brute force is sometimes effective.  Lets take a look at the data and see what groups are relevant, then scan for those groups.  Twitter is a common theme, so we can search for all twitter related projects with 
```python
for key in sumbags:
	if 'twitt' in key or 'tweet' in key:
		print(key)
output:tweet
twitter
readtweetapp
tweets
newstweet
tweeting
retweet
polytweet
tweetview
twitterverse
```
We are going to change the dataframe so that each of these project becomes simply 'twitter'.  The following funciton can achieve this with a manual input of keywords as the wordlist, and resets the project name to 'topic'.  The print statement is an internal check that is useful during development.

```python

#convert words in the project description to one key word = topics
topic_list={} #dictionary of topics with counts as the value of each topic key

def topic(topic,wordlist):
    count=0
    for project in x.project:
        for word in wordlist:
            if word in project.lower():
                count+=1
                x.topic[x.project==project]=topic
    topic_list[topic]=count

#define these lists based on observation.
#The function  'topic' takes a key word and a list as parameters
topic('money',['money','currency','loan','lend','stock','hedge','invest','financ','bank'])

topic('food',['food','yelp','bake','recipe','beer','wine','liquor','sandwich','cook',
              'delicious','cuisine','meal','yum','tasty','coffee','restaurant','dinner','lunch','breakfast','diet','nutrition'])

topic('social',['social','facebook','friend','romance','dating','people'])

topic('transportation',['flight','trip','traffic','travel','airplane','airport','airfare',
                        'delay','bnb','walk','route','vacation','transport','taxi'])

topic('twitter',['tweet','twitt'])

topic('eductation',['student','teacher','school','university','education'])

topic('media',['news','book','music','youtube','movie','song','media'])

topic('business',['customer','churn','career','job','business','business',
                  'startup','product','market','b2b','retail','shop'])

topic('sports',['nhl','nfl','nba','football','baseball','basketball','health','sports',
                'running','marathon','fitness','bike','hike','trainer'])
```

After running the program with these lists, we can iterate the process and add additional words to the topic list as needed. The following code run in the command line shows the projects that are not categorized

```python
for i in range(len(x)):
	if x.topic[i] not in topic_list:
		print(x.project[i])
```
Then, lets take a look at what are the most common topics by plotting a new histogram with the topics and their values 
```python
plt.bar([topic for topic in topic_list],[topic_list[topic] for topic in topic_list],color='g')
plt.xticks(rotation=90)
plt.tight_layout(pad=2)
plt.title('%d Most Common Topics in Insight Data Fellows Project Titles'%number)
plt.ylabel('Counts')
plt.show()
```

![Common Topics](https://github.com/jeffsecor/InsightFaces/blob/master/commonTopics.png)

These will be the target categories for our machine learning model.

## Clustering Faces

### Making the Dataframe and Serializing Images

The goal is now to input a group of faces as the data and the project topic as the target variable.  This is not like conventional facial clustering, because we are not trying to cluster images that are similar, but rather we need to find a metric (or metrics) that can be used to cluster each topic group.  Based on the work done above, we have about 350 assigned topics.  We will use these as our training/testing data.

First, save the faces to a location on disk using the **urllib** package. The **[-4:]** takes the file extension from the url and appends it to the filename for proper extension handling.  
```python
import urllib.request
import os
import re
i=0
path = "C:\\Users\\Ruddiger\\AppData\\Local\\Programs\\Python\\Python36\\Faces\\Images\\"
for url in url_list:
    try:
        urllib.request.urlretrieve(url,
                               path+'{}{}'.format(i,url[-4:]))
    except:
        print(i)
        pass
    i+=1
```
which outputs all the images, and the number 777 which means we need to make sure we associate number 777 in the data base to the entry with "No Image".  Lets take a look at some of the fellows in the money category
```python
import os
import cv2
from imutils import build_montages

path = 'C:\\Users\\Ruddiger\\AppData\\Local\\Programs\\Python\\Python36\\Faces\\Images'
money_list=x.index[x.project=='money'] #get values where topic is 'money'

#make list of images with topic == money
money_faces=[cv2.imread(os.path.join(path,'{}.jpg'.format(index))) for index in money_list] 
montage = build_montages(money_faces, (96, 96), (6, 6))[0]
cv2.imshow('money',montage)
```
<img src="https://github.com/jeffsecor/InsightFaces/blob/master/moneyfaces.PNG" width="512">
 
I will use the DBSCAN package for facial analysis. This captures only the facial region and serializes the data.  The following is modified  from https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/ . 
```python
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import re

# grab the paths to the input images in our dataset, then initialize
# out data list (which we'll soon populate)
print("[INFO] quantifying faces...")
path = 'C:\\Users\\Ruddiger\\AppData\\Local\\Programs\\Python\\Python36\\Faces\\Images\\'
imagePaths = list(paths.list_images(path))
data = []

# loop over the image paths
# load the input image and convert it from RGB (OpenCV ordering)
# to dlib ordering (RGB)
for (i, imagePath) in enumerate(imagePaths):
        print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
        print(imagePath)
        imagenumber=re.search(r'\d*\.',imagePath).group().strip('.')
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model="hog")

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
         
        # build a dictionary of the image path, bounding box location,
        # and facial encodings for the current image
        d = [{'ImagePath':imagePath,"PhotoNumber": imagenumber, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
        data.extend(d)

        # dump the facial encodings data to disk
        print("[INFO] serializing encodings...")
        f = open(path+'pickle', "wb")
        f.write(pickle.dumps(data))
        f.close()
```
The output is a pickle file that contains the image encodings along with the coordinates of the bounding box of the face.  We want to continue building our dataframe to include this information.  To do so, we can include two new column to our dataframe that will hold **arrays**, but calling a new column as **x['encodings']=pd.isnull**  The null is helpful because it does not set the type of object that goes into this column yet, so there will not be a conflict when an array is passed to the dataframe.

Since the order of the encodings does not match the order of the original dataframe, we need to scan each encoding within the pickle file and assign it to the proper index in our dataframe.  This can be achieved by matching photo number in the pickle object to the index in the original dataframe.  
```python
for d in data:
	num = int(d['PhotoNumber'])
	fellows.encoding.iloc[num]=d['encoding']
	fellows.box.iloc[num]=d['loc']
	fellows.path.iloc[num]=d['ImagePath']

```
This updates the data frame to include the encoding and bounding box location.  Finally, we save this dataframe with a save to csv
```python
fellows.to_csv('facebook.csv')
```

## Setting Targets and Executing the Cluster Model 
We want to assign the topic as the target for each fellow's image that has been classified.  Then we will make a cluster of each topic based on the encodings of all the images in the topic.

