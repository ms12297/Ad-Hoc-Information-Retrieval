import re
from nltk.stem import PorterStemmer as ps
import math # log functions
from scipy import spatial # cosine distance = 1 - cosine similarity

# ms12297-HW4

closed_class_stop_words = ['a','the','an','and','or','but','about','above','after','along','amid','among',\
                           'as','at','by','for','from','in','into','like','minus','near','of','off','on',\
                           'onto','out','over','past','per','plus','since','till','to','under','until','up',\
                           'via','vs','with','that','can','cannot','could','may','might','must',\
                           'need','ought','shall','should','will','would','have','had','has','having','be',\
                           'is','am','are','was','were','being','been','get','gets','got','gotten',\
                           'getting','seem','seeming','seems','seemed',\
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these', \
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',\
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',\
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',\
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',\
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',\
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',\
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',\
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace', \
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',\
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',\
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',\
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs',\
                           'you','your','yours','me','my','mine','I','we','us','much','and/or'
                           ]

unq_words = set()
idx_word = {}
index = 0
doc_ref = {}


def load(file_path):
    global index

    num_doc = 0 # IDF numerator - total number of texts 
    idf = {} # IDF denominator - dict storing word and number of documents containing the word pairs
    tf = [] # TF numerator - list of dicts: for each text, a word, freq pair is stored
    words = [] # TF denominator - list of number of words in each text
    doc_id = 1
    
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.rstrip().split()
            if line[0] == '.I': # record id
                curr_id = line[1]
                doc_ref[curr_id] = doc_id
                doc_id += 1
                continue
            elif line[0] == '.W': # text area, record content
                num_doc += 1
                num_words = 0
                word_curr = {}
                line = f.readline()
                if not line:
                    break
                line = line.rstrip().split()

                while line[0][0] != '.':
                    num_words += len(line)
                    for word in line:
                        if word not in closed_class_stop_words:
                            if word:
                                word = re.sub('[^a-z]', '', word) # remove non-alphabet
                                word = ps().stem(word) # stemming - higher MAP
                                try:
                                    word_curr[word] += 1
                                except KeyError:
                                    word_curr[word] = 1
                                if word not in unq_words:
                                    idx_word[index] = word
                                    index += 1
                                    unq_words.add(word)
                    line = f.readline()
                    if not line:
                        break
                    line = line.rstrip().split() 

                if not line:
                    tf.append(word_curr)
                    words.append(num_words)
                    for word in word_curr:
                        try:
                            idf[word] += 1
                        except KeyError:
                            idf[word] = 1
                    break
                if line[0] == '.I':
                    curr_id = line[1]
                    doc_ref[curr_id] = doc_id
                    doc_id += 1
                tf.append(word_curr)
                words.append(num_words)
                for word in word_curr:
                    try:
                        idf[word] += 1
                    except KeyError:
                        idf[word] = 1
            else:
                continue

    return idf, num_doc, tf, words


def write(num_q, num_abs, vec_q, vec_abs): #num_q, num_abs, vec_q, vec_abs
    out_file = open('output.txt', 'w', newline='\n')
    for i in range(1, num_q+1):
        # calc cosine similarity and append output line
        out = []
        for j in range(1, num_abs+1):
            similarity = 0
            if sum(vec_abs[j-1]) != 0:
                # cosine similarity = 1 - cosine distance
                similarity = 1 - spatial.distance.cosine(vec_q[i-1], vec_abs[j-1])
            out.append((similarity, j))
        # sorting
        out.sort(key=lambda x: x[0], reverse=True)
        # writing to file
        count = 0 
        for similarity, abs_idx in out:
            if count == 100: # output only top 100
                break
            #sim = str(similarity).replace('\r\r', '')
            line = str(i) + ' ' + str(abs_idx) + ' ' + str(similarity) + '\n'
            #line = line.rstrip()
            #line = line.replace('\r\r', '')
            # print(line)
            out_file.write(line)
            #out_file.write('\n')
            count += 1


if __name__ == '__main__':

    # load data
    q_idf, num_q, q_tf, q_words = load('cran.qry')
    abs_idf, num_abs, abs_tf, abs_words = load('cran.all.1400')

    #creating vectors
    vec_q = [[0 for x in range(index)] for x in range(num_q)]
    vec_abs = [[0 for x in range(index)] for x in range(num_abs)]

    # calculate tf-idf
    for j in range(index):
        for i in range(num_q): 
            tf = 0
            if q_words[i] != 0:
                try:
                    if q_tf[i][idx_word[j]]:
                        tf = math.log(q_tf[i][idx_word[j]] + 1)  # +1 for smoothing  
                except:
                    pass
            try:
                idf = math.log((num_q + 1) / (q_idf[idx_word[j]] + 1))
            except KeyError:
                idf = 0
            vec_q[i][j] = tf * idf

        for i in range(num_abs):
            tf = 0
            if abs_words[i] != 0:
                try:
                    if abs_tf[i][idx_word[j]]:
                        tf = math.log(abs_tf[i][idx_word[j]] + 1) 
                except:
                    pass
            try:
                idf = math.log((num_abs + 1) / (abs_idf[idx_word[j]] + 1))
            except KeyError:
                idf = 0
            vec_abs[i][j] = tf * idf

    # write to file
    write(num_q, num_abs, vec_q, vec_abs)
