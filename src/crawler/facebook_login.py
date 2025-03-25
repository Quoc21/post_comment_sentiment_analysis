import json
import os
from random import randint
from src.crawler.resources.css_selectors import CSS_SELECTOR
from src.crawler.crawler_utils import delay, is_cookie_expired
from src.config.config import CONFIG

def login_facebook_and_save_state(browser, username, password):
    context = browser.new_context()
    page = context.new_page()
    state_path = CONFIG['fb_state_path']
    page.goto('https://www.facebook.com/', timeout=120000)

    page.wait_for_selector(CSS_SELECTOR['id_email_input'])
    page.wait_for_selector(CSS_SELECTOR['id_pass_input'])
    delay(page)

    page.type(CSS_SELECTOR['id_email_input'], username, delay=randint(100, 300))
    delay(page)
    page.type(CSS_SELECTOR['id_pass_input'], password, delay=randint(100, 300))
    delay(page)
    page.click(CSS_SELECTOR['name_login_button'])

    page.wait_for_selector(CSS_SELECTOR['role_navigation_div'], timeout=120000)
    context.storage_state(path=state_path)

    return page

def login_facebook_with_state(browser):
    state_path = CONFIG['fb_state_path']
    context = browser.new_context(storage_state=state_path)
    page = context.new_page()
    page.goto('https://www.facebook.com/', timeout=120000)
    return page

def login_facebook(browser, username, password):
    state_path = CONFIG['fb_state_path']
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            state = json.load(f)
        cookies = state.get('cookies', [])

        if any(is_cookie_expired(cookie) for cookie in cookies):
            return login_facebook_and_save_state(browser, username, password)

        return login_facebook_with_state(browser)

    return login_facebook_and_save_state(browser, username, password)
