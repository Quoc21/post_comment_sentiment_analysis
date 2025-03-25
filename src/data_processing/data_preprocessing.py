import emoji
import re
import py_vncorenlp
from src.config.config import CONFIG
from underthesea import text_normalize
from src.utils.file_utils import load_data_json, save_data_json
from src.utils.data_processing_utils import get_metadata, count_comment

VNCORENLP_PATH = 'E:\Work\AI\Project\post_comment_sentiment_analysis\.venv\Lib\site-packages\py_vncorenlp'

def preprocess_comments(comments, rdrsegmenter, abbr, emoji_vi):
    if not comments:
        return

    for cmt in comments:
        cmt['cmt_content'] = preprocess_text(cmt['cmt_content'], rdrsegmenter, abbr, emoji_vi)
        if cmt['label'] is None:
            cmt['label'] = 2

        preprocess_comments(cmt['comments'], rdrsegmenter,abbr, emoji_vi)

def replace_emoji_and_abbr(text, abbr, emoji_vi):
    pattern = r':[^:\s]+:|[\w]+|[^\w\s]'
    words = re.findall(pattern, text)

    for i, w in enumerate(words):
        if w in abbr:
            words[i] = abbr[w]
        elif len(w) > 2 and w.startswith(':') and w.endswith(':'):
            emoji_vi_name = emoji_vi.get(w, '')
            if emoji_vi_name:
                words[i] = f'[{emoji_vi_name}]'
            else:
                words[i] = ''

    return ' '.join(words)

def preprocess_text(text, rdrsegmenter, abbr, emoji_vi):
    text_with_name_emoji = emoji.demojize(text)

    text_after_replace = replace_emoji_and_abbr(text_with_name_emoji, abbr, emoji_vi)

    normalized_text = text_normalize(text_after_replace)

    segmented_text = ' '.join(rdrsegmenter.word_segment(normalized_text))

    return segmented_text

def remove_empty_cmt(comments, level):
    new_comments = []
    for comment in comments:
        children = remove_empty_cmt(comment.get('comments', []), level + 1)

        content = comment.get('cmt_content') or ""

        if content.strip() == "":
            if level < 3:
                new_comments.extend(children)
        else:
            comment['comments'] = children
            new_comments.append(comment)
    return new_comments

def preprocess_data(data_path, return_data=False, file_path='preprocessed_data.json'):
    abbr = load_data_json(CONFIG['abbreviation_path'])
    emoji_vi = load_data_json(CONFIG['emoji_path'])
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=VNCORENLP_PATH)

    data = load_data_json(data_path)

    for post in data['posts']:
        post['post_content'] = preprocess_text(post['post_content'], rdrsegmenter, abbr, emoji_vi)
        preprocess_comments(post['comments'], rdrsegmenter, abbr, emoji_vi)
        post['comments'] = remove_empty_cmt(post.get('comments', []), level=1)
        post['cmt_count'] = count_comment(post['comments'])

    data['metadata'] = get_metadata(data['posts'])

    if return_data:
        return data

    save_data_json(data, file_path)