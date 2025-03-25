import os
import json
import random
from datetime import datetime
from src.config.config import CONFIG
def delay(page):
    time_to_wait = random.uniform(1000, 1500)
    page.wait_for_timeout(time_to_wait)

def is_cookie_expired(cookie):
    if 'expires' in cookie:
        expires = datetime.fromtimestamp(cookie['expires'])
        return expires < datetime.now()
    return False

def save_post(post, path, indent=4, ensure_ascii=False):
    if not os.path.exists(path):
        data = {
            'metadata': {
                'post_count': 1,
                'cmt_count': post['cmt_count'],
                'pos_count': 0,
                'neg_count': 0,
                'neu_count': 0
            },
            'posts': [post]
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data['metadata']['post_count'] += 1
        data['metadata']['cmt_count'] += post['cmt_count']
        data['posts'].append(post)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

def post_exists(post_url):
    path = CONFIG['raw_data_path']
    if not os.path.exists(path):
        return False

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    posts = data['posts']
    for post in posts:
        if post['post_url'] == post_url:
            return True

    return False

def save_html(html):
    with open('post.html', 'w', encoding='utf-8') as f:
        f.write(html)

def load_html():
    with open('post.html', 'r', encoding='utf-8') as f:
        html = f.read()
    return html
