from src.config.config import CONFIG
from src.crawler.data_crawling import crawl_post_data
from src.data_processing.data_preprocessing import preprocess_data

if __name__ == '__main__':
    # region preprocess data

    raw_data_path = CONFIG['raw_data_path']
    preprocess_data(raw_data_path)

    # endregion

    # region crawl data
    # Nhớ xoá thông tin khi push code
    # username = ''
    # password = ''
    # post_url = 'https://www.facebook.com/share/p/1AbVq4VtFy/'
    # crawl_post_data(username, password, post_url)
    # endregion

    pass