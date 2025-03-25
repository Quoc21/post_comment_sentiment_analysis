from playwright.sync_api import sync_playwright
from random import uniform
from src.crawler.resources.css_selectors import CSS_SELECTOR
from src.crawler.crawler_utils import delay, post_exists, save_post
from src.crawler.facebook_login import login_facebook
from src.crawler.post_content_extracting import get_post_data
from src.config.config import CONFIG

def click_all_view_reply_buttons(page):
    while True:
        view_reply_buttons = page.locator(CSS_SELECTOR['class_view_reply_button'])
        button_count = view_reply_buttons.count()
        if button_count == 0:
            break
        view_reply_buttons.nth(0).click()
        delay(page)

def click_all_view_more_buttons(page):
    while True:
        view_more_buttons = page.get_by_role('button', name='Xem thêm')
        button_count = view_more_buttons.count()
        if button_count == 0:
            break
        view_more_buttons.nth(0).click(force=True)
        delay(page)

def scroll_to_load_comments(page):
    # Di chuột vào giữa để cuộn
    viewport_width = 1280
    viewport_height = 800
    center_x = viewport_width // 2
    center_y = viewport_height // 2
    page.mouse.move(center_x, center_y)

    body_post = page.locator(CSS_SELECTOR['class_body_post'])
    old_height, new_height = 0, body_post.bounding_box()['height']
    while old_height != new_height:
        # Nhấn các nút xem phản hồi hiện có
        click_all_view_reply_buttons(page)
        # Nhấn các nút xem thêm có
        click_all_view_more_buttons(page)

        # Cuộn xuống
        px_to_scroll = uniform(2000, 3000)
        page.mouse.wheel(0, px_to_scroll)
        old_height = new_height
        new_height = body_post.bounding_box()['height']
        delay(page)

    return body_post

def crawl_html(page, post_url):
    page.goto(post_url, timeout=120000, wait_until='domcontentloaded')
    delay(page)

    # Thao tác xổ tất cả bình luận
    most_popular_button = page.locator(CSS_SELECTOR['class_most_popular'])
    most_popular_button.scroll_into_view_if_needed()
    most_popular_button.click()
    delay(page)
    page.locator(CSS_SELECTOR['class_most_popular_menu']).nth(2).click()
    delay(page)

    post_full_content = scroll_to_load_comments(page)

    return post_full_content.inner_html()

def crawl_post_data(username, password, post_url, return_post=False, headless=False):
    if post_exists(post_url):
        print('Post already exits !!!')
    else:
        print('Crawling...')
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=headless)
                page = login_facebook(browser, username, password)
                html = crawl_html(page, post_url)
                browser.close()

            post = get_post_data(html)
            post['post_url'] = post_url
            path = CONFIG['raw_data_path']
            if return_post:
                return post
            else:
                save_post(post, path=path)
                print('Yeah !!!')
        except Exception as e:
            print(e)

    pass