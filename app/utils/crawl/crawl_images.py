from CrawlData.searchByImage.search_by_image import search_by_image
from CrawlData.searchByText.search_by_key import search_by_text


def Crawl_urls_data(image_path: str = None, text: str = None):
    if image_path:
        return search_by_image(image_path)
    elif text:
        return search_by_text(text)
    else:
        return "Please provide either image_path or text"