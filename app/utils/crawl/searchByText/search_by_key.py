from CrawlData.searchByText.Sougou_search import SogouImageScraper
import concurrent.futures
import time


def search_by_text_Sougou(query, num_images=1000):
    return SogouImageScraper(num_images=num_images).fetch_image_urls(query)

def run_with_retry(func, args, max_retries=3):
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            result = func(*args)
            end_time = time.time()
            duration = end_time - start_time
            print(f"{func.__name__} succeeded in {duration:.2f} seconds on attempt {attempt+1}")
            return result
        except Exception as e:
            print(f"{func.__name__} failed with {e}, retrying ({attempt+1}/{max_retries})...")
            time.sleep(1)  
    raise Exception(f"{func.__name__} failed after {max_retries} attempts")

def search_by_text(keyword):
    URLS = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(run_with_retry, search_by_text_Sougou, (keyword,)): 'Sougou',
        }

        for future in concurrent.futures.as_completed(futures):
            source = futures[future]
            try:
                urls = future.result()
                URLS.extend(urls)
            except Exception as exc:
                print(f"{source} generated an exception: {exc}")

    print(f"Total URLs collected: {len(URLS)}")
    return URLS
