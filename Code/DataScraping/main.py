import requests
from bs4 import BeautifulSoup

def getAllLinks():
    global newsLinks
    i = 1
    while len(newsLinks) < 1000:
        page = requests.get("https://www.tin247.news/cong-nghe-4-"+str(i)+".tintuc")

        soup = BeautifulSoup(page.content, "html.parser")

        newsArr = soup.find_all("li", {"class": "_box item"})

        for news in newsArr:
            newsLinks.append("https://www.tin247.news"+news.find('a', href = True)['href'])

        i += 1


def getAllContent():
    global newsLinks

    for i in range(len(newsLinks)):
        page = requests.get(newsLinks[i])
        soup = BeautifulSoup(page.content, "html.parser")
        detail = soup.find("section", {"class": "_box news_detail"})
        title = detail.find("h1", {"class": "title"}).text
        title = ' '.join(e for e in title.split(" ") if e.isalnum())
        print(title)
        fullText = ""
        for t in detail.find_all("p"):
            fullText += "\n" + t.text

        with open("./data/"+title+".txt", "w+", encoding="utf-8") as f:
            f.write(fullText)

if __name__ == "__main__":
    newsLinks = []

    getAllLinks()

    getAllContent()