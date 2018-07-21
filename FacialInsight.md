# Predicting Project Themes Based on Facial Features of Insight Data Science Fellows

The following will be a journal style entry as I create the program. There will also be another file with the self-contained python code.  The question we are after is "can we predict the type of project a fellow will choose based on their facial features?" My intial guess is 'no' because I suspect that the common characteristic of Insight fellows is their knowledge and technical methods, not their appearance.  But, it seems like a fun question to ask, and also presents a very in depth learning experience for a novice programmer such as myself.  Im learning as I go, so this may get a little bumpy.

The general plan is:
1. Scrape image files and project text from the webpage
2. Build database architecture for images with corresponding text
3. Identify classes/groups/types of projects
4. Learn facial recognition packages and identify facial characteristics
5. Look at correlations of selected facial features to projects

## 1. Scraping images and corresponding project text
As one would expect from a data science fellowship program, the Insight website is well written and the page source is dense and complex, but easy(ish) to navigate.  To start, we will pull the webpage using beautiful soup

```python
import bs4  # BeautifulSoup4 to scrape the webpage#
url = 'https://www.insightdatascience.com/fellows'
response=requests.get(url)
html=response.content
soup=bs4.BeautifulSoup(html,'lxml')
```

Looking at the page source (left click and 'view source') shows us how to find the images.  Each fellows image exists inside a div class 'fellow_img' (Im new to html, so apologies for any incorrect terminology).  The let's make a list containing all the image urls.  We use beautiful soup to find all the 'fellow_img' class data within 'div' tags
```python
data=[info for info in soup.find_all(class_= "fellow_img")]
```
The size of our list can be found with **len(data)** which tells us that there are 794 entries.  Lets take a look at one of them with **print(data[5])** which outputs
```python
<div class="fellow_img" style='background-image:url("https://daks2k3a4ib2z.cloudfront.net/575a31d2ce5d01dc7a20de45/5769bb9322c61e320363317d_Jessica-Pic.jpg")'><div class="fellow_caption_back"><div class="fellow_name">Jessica Zúñiga</div><div class="fellow_company">LinkedIn</div></div></div> 
```
The url is in the 'background image' style.  It took a bit for me to figure this out.  One thing that helped is to check the type of this thing by **type(data[5])** which tells us that it is a 'bs4.element.Tag'.  That means we can use some function of beautiful soup to extract the url information.  A bit of searching leads to the right sytax, which is pleasantly intuitive.  The text of the style section is pulled with **data[5]["style"]** which outputs 'background-image:url("https://daks2k3a4ib2z.cloudfront.net/575a31d2ce5d01dc7a20de45/5769b885e8628b5103c73621_davidfreeman-main.jpg")'.  Checking **type(data[5]['style'])** tells us that it is a string , which means we can use regex to pull out the url.  Since they are all http....jpg, the regex is straightforward.  Except the group function is needed to output the match as a string instead of a regex match result.  

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
This works to pull out the url from the first element in the list **data**.  The next thing we want is to cath all the urls from the soup.  I found that the **.group()** was giving an error becuase the **.group()** function could not act on a type "none".  Checking the documentation of the regular expression functions, 'none' is returned when there is not a match, and so there is no string to group together.  To find the root of the problem, I printed each match as it came out in the loop, along with the text it was trying to match, and found that some images are png, while others are JPG, PNG, or in some cases there are no images.  To work around this,  we can change our matching conditions to be case insensitive using **(i?)** and also an 'or' condition.  The matching expression now looks like 
```python
match=re.search(r'https.*jpg(?i)|png(?i)',text).group()
```
The missing images will also throw an error from the **.group()** function, so a **try...except** structure is appropriate.  We make a new list of our image urls and run a for loop.  The print statements at the end are to check to fidelity of the data and regex
```python
url_list=[]
for i in range(len(data)):
    try:
        url_list.append(re.search(r'https.*jpg(?i)|png(?i)',data[i]['style']).group())
    except:
        url_list.append('No Image')
print(len(data))  #check length of data list
print(len(url_list))  #check this list is the same length
print(data[402],"\n\n",url_list[402],"\n") #compare entries of each list
print(data[-1],"\n\n",url_list[-1],"\n")
```
There we have it, a list of image urls ready to be called on to download the images.  

The next step in the data collection and cleaning is to get the name of the projects that correspond to the headshots. Similar to how we found the image urls, we look at the webpage source and see that the project title is listed under a class 'tootip_project'.  This scraping step is a bit simpler beacause the information is the only text in the tags, so a single list comprehension with the **.text** function will work.  And check that the project list has the same length as the url_list.  Finally, double check a few values against the webpage to make sure the projects correspond to the right peoples faces
```python
projects=[project.text for project in soup.find_all(class_= "tooltip_project")]
```
And thats it for the project list, pretty simple there.


