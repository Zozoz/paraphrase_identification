
# coding: utf-8

# In[1]:

import numpy as np
from nltk.tokenize import word_tokenize


# In[2]:

def rewrite_data(src_file, dest_file):
    # toker = StanfordTokenizer()
    toker = word_tokenize
    sf = open(src_file)
    df = open(dest_file, 'w')
    sf.readline()
    cnt = 0
    for line in sf:
        cnt += 1
        line = line.decode('utf8', 'ignore')
        line = line.split('","')
        print line
        line[0] = line[0][1:]
        line[-1] = line[-1][0]
        line[3] = ' '.join(toker(line[3]))
        line[4] = ' '.join(toker(line[4]))
        # line[3] = ' '.join(toker.tokenize(line[3]))
        # line[4] = ' '.join(toker.tokenize(line[4]))
        # print line
        print line
        new_line = ' || '.join(line[1:]).encode('utf8', 'ignore') + '\n'
        print new_line
        df.write(new_line)
        # break
        if cnt % 1000 == 0:
            print cnt


# In[3]:

# rewrite_data('data/train.csv', 'data/train.txt')


# In[4]:

def rewrite_test_data(src_file, dest_file):
    toker = word_tokenize
    sf = open(src_file)
    df = open(dest_file, 'w')
    sf.readline()
    cnt = 0
    for line in sf:
        cnt += 1
        line = line.decode('utf8', 'ignore')
        line = ','.join(line.split(',')[1:]).split('","')
        print line
        line[0] = ' '.join(toker(line[0][1:]))
        line[1] = ' '.join(toker(line[1][:-2]))
        print line
        new_line = ' || '.join(line).encode('utf8', 'ignore') + '\n'
        print new_line
        df.write(new_line)
        # break
        if cnt % 1000 == 0:
            print cnt


# In[5]:

# rewrite_test_data('data/test.csv', 'data/origin_test.txt')


# In[8]:

def load_train_test(src_file, word2id, max_sen_len, freq=5):
    sf = open(src_file)
    x1, x2, len1, len2, y = [], [], [], [], []
    def get_q_id(q):
        i = 0
        tx = []
        for word in q:
            if i < max_sen_len and word in word2id:
                tx.append(word2id[word])
                i += 1
        tx += ([0] * (max_sen_len - i))
        return tx, i
    for line in sf:
        line = line.lower().split(' || ')
        q1 = line[2].split()
        q2 = line[3].split()
        is_d = line[4][0]
        tx, l = get_q_id(q1)
        x1.append(tx)
        len1.append(l)
        tx, l = get_q_id(q2)
        x2.append(tx)
        len2.append(l)
        y.append([0, 1] if is_d == '0' else [1, 0])
    index = range(len(y))
    np.random.shuffle(index)
    x1 = np.asarray(x1, dtype=np.int32)
    x2 = np.asarray(x2, dtype=np.int32)
    len1 = np.asarray(len1, dtype=np.int32)
    len2 = np.asarray(len2, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)
    s = int(len(y) * 0.8)
    return x1[:s], x2[:s], len1[:s], len2[:s], y[:s], x1[s:], x2[s:], len1[s:], len2[s:], y[s:]


def load_sentence(src_file, word2id, max_sen_len, freq=5):
    sf = open(src_file)
    x1, x2, len1, len2, y = [], [], [], [], []
    def get_q_id(q):
        i = 0
        tx = []
        for word in q:
            if i < max_sen_len and word in word2id:
                tx.append(word2id[word])
                i += 1
        tx += ([0] * (max_sen_len - i))
        return tx, i
    for line in sf:
        line = line.lower().split(' || ')
        q1 = line[2].split()
        q2 = line[3].split()
        is_d = line[4][0]
        tx, l = get_q_id(q1)
        x1.append(tx)
        len1.append(l)
        tx, l = get_q_id(q2)
        x2.append(tx)
        len2.append(l)
        y.append([0, 1] if is_d == '0' else [1, 0])
    index = range(len(y))
    # np.random.shuffle(index)
    x1 = np.asarray(x1, dtype=np.int32)
    x2 = np.asarray(x2, dtype=np.int32)
    len1 = np.asarray(len1, dtype=np.int32)
    len2 = np.asarray(len2, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)
    return x1, x2, len1, len2, y

# In[7]:

def load_word_embedding(embed_file, embedding_dim, is_skip=True):
    fp = open(embed_file)
    if is_skip:
        fp.readline()
    w2v = []
    word2id = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    for line in fp:
        cnt += 1
        # line = line.decode('utf8').split()
        line = line.split()
        if len(line) != embedding_dim + 1:
            print u'a bad word embedding: {}'.format(line[0])
            continue
        w2v.append([float(v) for v in line[1:]])
        word2id[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    # w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    print np.shape(w2v)
    # word_dict['$t$'] = (cnt + 1)
    w2v -= np.mean(w2v, axis=0)
    w2v /= np.std(w2v, axis=0)
    return word2id, w2v


# In[ ]:

def cut_embedding(src_file, big_embed_file, small_embed_file, is_skip=True):
    fp = open(big_embed_file)
    w2v = dict()
    for line in fp:
        line = line.split()
        w2v[line[0]] = ' '.join(line[1:])
    fp.close()
    print 'load big embedding done!'

    words = set()
    sf = open(src_file)
    for line in sf:
        words.update(line.lower().split())
    sf.close()
    print 'load words done!'

    df = open(small_embed_file, 'w')
    for word in words:
        if word in w2v:
            df.write(word + ' ' + w2v[word] + '\n')
    print 'cut embedding done!'



# In[ ]:

# cut_embedding('data/all.txt', '/Users/newbie/Downloads/Corpus/glove/glove.840B.300d.txt', 'embedding_840b_300d.txt', False)


# In[1]:

def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    index = range(length)
    for j in xrange(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in xrange(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]


# In[ ]:




# In[ ]:



