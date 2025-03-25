def get_metadata(posts):
    metadata = {}

    cmt_count = 0
    for p in posts:
        cmt_count += p['cmt_count']

    metadata['post_count'] = len(posts)
    metadata['cmt_count'] = cmt_count

    post_count, neg_count, neu_count = count_label(posts)
    metadata['pos_count'] = post_count
    metadata['neg_count'] = neg_count
    metadata['neu_count'] = neu_count

    return metadata

def count_label_in_comments(comments):
    pos_count = 0
    neg_count = 0
    neu_count = 0

    if not comments:
        return pos_count, neg_count, neu_count

    for cmt in comments:
        if cmt['label'] == 0:
            neg_count += 1
        elif cmt['label'] == 1:
            pos_count += 1
        elif cmt['label'] == 2:
            neu_count += 1

        p, n, u = count_label_in_comments(cmt['comments'])
        pos_count += p
        neg_count += n
        neu_count += u

    return pos_count, neg_count, neu_count

def count_label(posts):
    pos_count = 0
    neg_count = 0
    neu_count = 0

    for post in posts:
        p, n, u = count_label_in_comments(post['comments'])
        pos_count += p
        neg_count += n
        neu_count += u

    return pos_count, neg_count, neu_count

def count_comment(comments):
    if not comments:
        return 0

    cmt_count = 0
    for cmt in comments:
        cmt_count += 1
        cmt_count += count_comment(cmt['comments'])
    return cmt_count