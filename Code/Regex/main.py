import requests
import re


def cleanHTML(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def getURL(raw_html):
    reg = r'(?:https?://\S+)'
    print(re.findall(reg, raw_html))


def getPhoneNum(raw_html):
    reg = r'([\+84|84|0]+[3|5|7|8|9|1])+([0-9]{8})\b'
    group = re.findall(reg, raw_html)
    for g in group:
        print("".join(g))


def getEmail(raw_html):
    reg = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    print(re.findall(reg, raw_html))


if __name__ == "__main__":
    page = requests.get(
        "https://www.tin247.news/nen-mua-iphone-14-vna-chinh-hang-hay-hang-xach-tay-hon-4-29475799.html")
    raw_html = page.content.decode("utf-8")
    raw_html += "pvtnguyet.19it1@vku.udn.vn 0941257069 +84941257069"

    print(raw_html)
    print(cleanHTML(raw_html))

    getURL(raw_html)
    getEmail(raw_html)
    getPhoneNum(raw_html)
