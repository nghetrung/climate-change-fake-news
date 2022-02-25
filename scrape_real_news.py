import requests
from bs4 import BeautifulSoup

# example
article = requests.get('https://www.newscientist.com/article/2238118-electric-cars-really-are-a-greener-option-than-fossil-fuel-vehicles/')
article_content = article.content
soup_article = BeautifulSoup(article_content)
body = soup_article.find_all('div', class_='article-content')
x = body[0].find_all('p', class_=None)

news_content = []
list_paragraphs = []
for i in range(0, len(x)):
    paragraph = x[i].get_text()
    if paragraph[0].isalnum():
        list_paragraphs.append(paragraph)

final_article = " ".join(list_paragraphs)
news_content.append(final_article)


# The Guardian
news_contents = [] # final list of articles
n_pages = 25
page_link = 'https://www.theguardian.com/environment/climate-change'
for i in range(n_pages):
    page = requests.get(page_link)
    page_content = page.content
    soup_page = BeautifulSoup(page_content)
    articles = soup_page.find_all('a', class_='fc-item__link')
    for j in range(len(articles)):
        article_link = articles[j]['href']
        print(article_link)
        article = requests.get(article_link)
        article_content = article.content
        soup_article = BeautifulSoup(article_content)
        headline = soup_article.find('h1', class_='content__headline')
        print(headline)
        body = soup_article.find_all('div', class_='content__article-body from-content-api js-article__body')
        print(len(body))
        # some article dont have text (e.g. podcast, pictures only)
        if len(body) > 0: 
            paragraphs = body[0].find_all('p')
            list_paragraphs = []
            for p in range(len(paragraphs)):
                paragraph = paragraphs[p].get_text()
                list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)
            news_contents.append(final_article)
    # get next page link
    page_link = soup_page.find('a', class_='button button--small button--tertiary pagination__action--static', rel='next')['href']
    print(f'Going to next page...', page_link)



# NASA: Global Climate Change
from selenium import webdriver
import time

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
# options.add_argument('--headless')
driver = webdriver.Chrome("C:/Users/nghet/chrome_driver/chromedriver.exe", chrome_options=options)

driver.get("https://climate.nasa.gov/news/?page=0&per_page=40&order=publish_date+desc%2C+created_at+desc&search=&category=19%2C98")
# click more 10 times => 11*40 = 440 articles
for i in range(10):
    more_button = driver.find_elements_by_xpath("//*[@id='page']/section[2]/div/div/section/div/article/footer/a")
    more_button[0].click()
    time.sleep(5)
page_source = driver.page_source
soup = BeautifulSoup(page_source)
articles_nasa = soup.find_all('li', class_='slide')
len(articles_nasa)

news_contents_nasa = []
host = 'https://climate.nasa.gov'
for i in range(len(articles_nasa)):
    article_link = host + articles_nasa[i].find('a')['href']
    print(article_link)
    article = requests.get(article_link)
    article_content = article.content
    soup_article = BeautifulSoup(article_content)
    headline = soup_article.find('h1', class_='article_title')
    print(headline)
    body = soup_article.find_all('div', class_='wysiwyg_content')
    print(len(body))
    # some article dont have text (e.g. podcast, pictures only)
    if len(body) > 0: 
        paragraphs = body[0].find_all('p')
        list_paragraphs = []
        for p in range(len(paragraphs)):
            paragraph = paragraphs[p].get_text()
            list_paragraphs.append(paragraph)
        final_article = " ".join(list_paragraphs)
        news_contents_nasa.append(final_article)

news_contents = news_contents + news_contents_nasa

# save real news to json
import json
news = {}
for index, new in enumerate(news_contents):
    news['train-'+str(index)] = {'text': new, 'label': 0}

with open('train_real_test.json', 'w') as r:  
    json.dump(news, r)


# livescience.com
news_contents_livescience = [] # final list of articles
n_pages = 20
page_link = 'https://www.livescience.com/topics/climate-change'
for i in range(n_pages):
    page = requests.get(page_link)
    page_content = page.content
    soup_page = BeautifulSoup(page_content)
    articles_list = soup_page.find('div', class_='listingResults')
    articles_links = articles_list.find_all('a', class_='article-link')
    for j in range(len(articles_links)):
        print('article:', j)
        article_link = articles_links[j]['href']
        print(article_link)
        article = requests.get(article_link)
        article_content = article.content
        soup_article = BeautifulSoup(article_content)
        main_article = soup_article.find('article', class_='news-article')
        headline = main_article.find('h1').get_text()
        print(headline)
        body = main_article.find_all('div', class_='text-copy bodyCopy auto')
        print(len(body))
        # some article dont have text (e.g. podcast, pictures only)
        if len(body) > 0: 
            paragraphs = body[0].find_all(['p', 'li'])
            list_paragraphs = []
            for p in range(len(paragraphs)):
                paragraph = paragraphs[p].get_text()
                list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)
            news_contents_livescience.append(final_article)
    # get next page link
    page_navigate = soup_page.find('span', class_='listings-pagination-button listings-next')
    page_link = page_navigate.find('a')['href']
    print(f'Going to next page...', page_link)




# load the real news to add more news
with open('train_real.json', 'r') as f:
    real_news = json.load(f)

append_index = len(real_news)
for index, new in enumerate(news_contents_livescience):
    real_news['train-'+str(index+append_index)] = {'text': new, 'label': 0}

with open('train_real_1.json', 'w') as r:  
    json.dump(real_news, r)