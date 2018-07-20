# Predicting Project Themes Based on Facial Features of Insight Data Science Fellows

The following will be a journal style entry as I create the program. There will also be another file with the self-contained python code.  The question we are after is "can we predict the type of project a fellow will choose based on their facial features?" My intial guess is 'no' because I suspect that the common characteristic of Insight fellows is their knowledge and technical methods, not their appearance.  But, it seems like a fun question to ask, and also presents a very in depth learning experience for a novice programme such as myself.  Im learning as I go, so this may get a little bumpy.

The overall plan is:
1. Scrape image files project text from the webpage
2. Build database architecture for images with corresponding text
3. Identify classes/groups/types of projects
4. Learn facial recognition packages and identify facial characteristics
5. Look at correlations of selected facial features to projects

## 1. Scraping images and corresponding project text
As one would expect from a data science fellowship program, the Insight website is well written and the page source is dense and complex, but easy(ish) to navigate.  To start, we will pull the webpage using beautiful soup

```python
url = 'https://www.insightdatascience.com/fellows'
response=requests.get(url)
html=response.content
soup=bs4.BeautifulSoup(html,'lxml')
```
