import requests
from bs4 import BeautifulSoup

from .utils import concat_paths

FAQ_CATEGORIES = {
    "BDC FAQs": "60000157358",
    "BDC Fellows Program FAQs": "60000294708",
}
FRESHDESK_BASE_URL = "https://bdcatalyst.freshdesk.com"


def extract_folder_links(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        hrefs, titles = [], []
        for section in soup.find_all("section", class_="cs-g article-list"):
            for div in section.find_all("div", class_="list-lead"):
                link = div.find("a")
                if link and "href" in link.attrs:
                    hrefs.append(link["href"])
                    titles.append(link.get("title", ""))
        return hrefs, titles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return [], []


def extract_article_links(url):
    try:
        response = requests.get(url, timeout=30)
        soup = BeautifulSoup(response.text, "html.parser")
        section = soup.find("section", class_="article-list c-list")
        hrefs = []
        if section:
            for row in section.find_all("div", class_="c-row c-article-row"):
                link = row.find("a", class_="c-link")
                if link and link.get("href"):
                    hrefs.append(link["href"])
        return hrefs
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []


def extract_text_from_html(url):
    try:
        response = requests.get(url, timeout=30)
        soup = BeautifulSoup(response.text, "html.parser")
        article_body = soup.find("article", class_="article-body")
        body_text = article_body.get_text(strip=True) if article_body else ""
        heading = soup.find("h2", class_="heading")
        title = heading.find(string=True, recursive=False).strip() if heading else ""
        return body_text, title
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return "", ""


def scrape_freshdesk(category_dict=None, base_url=FRESHDESK_BASE_URL):
    category_dict = category_dict or FAQ_CATEGORIES
    url_category_path = "/support/solutions"

    folder_url_list, folder_metadata_list = [], []
    for category, folder_id in category_dict.items():
        hrefs, titles = extract_folder_links(concat_paths(base_url, url_category_path, folder_id))
        folder_url_list.extend(hrefs)
        folder_metadata_list.extend([{"category": category, "folder": title} for title in titles])

    article_url_list, article_metadata_list = [], []
    for url, metadata in zip(folder_url_list, folder_metadata_list):
        hrefs = extract_article_links(concat_paths(base_url, url))
        article_url_list.extend(hrefs)
        article_metadata_list.extend([metadata] * len(hrefs))

    article_content_list = []
    for i, (url, metadata) in enumerate(zip(article_url_list, article_metadata_list)):
        body_text, title = extract_text_from_html(concat_paths(base_url, url))
        article_content_list.append(body_text)
        article_metadata_list[i] = dict(metadata, title=title, page_url=concat_paths(base_url, url))

    return article_content_list, article_metadata_list
