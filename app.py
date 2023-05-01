from flask import Flask, Response, request, send_file
from flask import jsonify
import sqlite3
import re
import pickle
import yake
from keybert import KeyBERT
from summarizer import TransformerSummarizer
from transformers import pipeline
import warnings
import en_core_web_lg
from gensim.models import Word2Vec
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
import string
import numpy as np
import os
from transformers import GPT2Tokenizer
import spacy
import io
from wordcloud import WordCloud
import sys
import matplotlib.pyplot as plt
import torch
import nltk
import itertools
from nltk.stem.porter import *
import ktrain
from ktrain import text


from transformers import GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("model/")
model = GPT2LMHeadModel.from_pretrained("model/")


def token_tensor(text, model):
    indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens])
    model.eval()

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

    return predicted_text


def choose_from_top(probs, n=1):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # Normalize
    choice = np.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)


INDEXDIR = 'mesoc_index'
parser = en_core_web_lg.load(disable=["tagger", "ner"])
parser.max_length = 7000000

punctuations = string.punctuation

stopwords = list(STOP_WORDS)
custom_stop_words = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure',
    'rights', 'reserved', 'permission', 'used', 'using', 'arxiv', 'i.e.', 'license', 'fig', 'fig.',
    'al.', 'Elsevier', 'PMC', 'CZI', 'www'
]


def rank_docs(model, query, df, num):
    # [(paper_id, processed_abstract, url, cosine_sim)]
    cosine_list = []

    a = []
    query = query.split(" ")
    for q in query:
        try:
            a.append(model[q])
        except:
            continue

    for index, row in df.iterrows():
        centroid = row['centroid']
        total_sim = 0
        for a_i in a:
            cos_sim = np.dot(a_i, centroid) / (np.linalg.norm(a_i) * np.linalg.norm(centroid))
            total_sim += cos_sim
        cosine_list.append((row['title'], total_sim))

    cosine_list.sort(key=lambda x: x[1], reverse=True)  ## in Descedning order

    papers_list = []
    for item in cosine_list[:num]:
        papers_list.append((item[0], item[1]))
    return papers_list


def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stopwords and word not in punctuations]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


def check(sentence, words):
    res = [any([k in s for k in words]) for s in sentence]
    return [sentence[i] for i in range(0, len(res)) if res[i]]


def convertTuple(tup):
    stx = ''.join(tup)
    return stx


sql_data = 'db/mesoc_db.db'
conn = sqlite3.connect(sql_data, check_same_thread=False)


def extract_study_technique(text):
    '''Extracts the type of study design in paper.
    '''
    study_designs = [
        'case control',
        'case study',
        'cross sectional',
        'cross-sectional',
        'descriptive study',
        'ecological regression',
        'experimental study',
        'meta-analysis',
        'non-randomized',
        'non-randomized experimental study',
        'observational study',
        'prospective case-control',
        'prospective cohort',
        'prospective study',
        'randomized',
        'randomized experimental study',
        'randomised controlled trial',
        'retrospective cohort',
        'retrospective study',
        'simulation',
        'systematic review',
        'time series analysis',
    ]

    return [design for design in study_designs if design in text]


c = conn.cursor()


def get_text(id, lower=True):
    c.execute('SELECT   body_text  FROM paper WHERE ID_Doc=' + str(id))
    try:
        text = str(c.fetchone())
        if lower:
            text = text.lower()
        text = text.strip()
        text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
        text = re.sub(' +', ' ', text)
        text = re.sub(r"\d", "", text)
        text = text.strip()
    except:
        return jsonify({'msg': 'No data found!'}), 401
    return text


def get_text_domain(id, lower=True):
    c.execute('SELECT   sentence  FROM function WHERE Domain=' + str(id)+ ' and length(sentence) >50')
    try:
        text=''
        rows = str(c.fetchall())
        for row in rows :
            text =text+'. '+row
        if lower:
            text = text.lower()
        text = text.strip()
        text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
        text = re.sub(' +', ' ', text)
        text = re.sub(r"\d", "", text)
        text = text.strip()
    except:
        return jsonify({'msg': 'No data found!'}), 401
    return text

def get_text_both(id,id1,lower=True):
    c.execute('SELECT   body_text  FROM paper WHERE Social_impact=' + str(id) + ' and Cultural_domain=' + str(id1) + 'and length(body_text) >50')
    try:
        text = str(c.fetchone())
        if lower:
            text = text.lower()
        text = text.strip()
        text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
        text = re.sub(' +', ' ', text)
        text = re.sub(r"\d", "", text)
        text = text.strip()
    except:
        return jsonify({'msg': 'No data found!'}), 401
    return text


def get_text_impact(id, lower=True):
    c.execute('SELECT   body_text  FROM paper WHERE Social_impact=' + str(id)+ 'and length(body_text) >50')
    try:
        text = str(c.fetchone())
        if lower:
            text = text.lower()
        text = text.strip()
        text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
        text = re.sub(' +', ' ', text)
        text = re.sub(r"\d", "", text)
        text = text.strip()
    except:
        return jsonify({'msg': 'No data found!'}), 401
    return text

all_keywords = []

app = Flask(__name__)

if __name__ == '__main__':
    app.run()

@app.route('/')
def hello_world():
    return 'Mesoc api!'




@app.route('/study/<int:id>', methods=['GET'])
def get_study(id):
    c = conn.cursor()
    c.execute('SELECT abstract , body_text  FROM paper WHERE ID_Doc=' + str(id))
    try:
        x = str(c.fetchone())
        z = (str(extract_study_technique(x)).replace("[", "").replace("]", "").replace("'", ""))

        return Response(response=str(z), status=200, mimetype="application/json")
    except:
        return Response("No data found.", status=500, mimetype="application/json")


@app.route('/keywords_y/<int:id>', methods=['GET'])
def get_keywords_y(id):
    with open('keywords.pkl', 'rb') as fp:
        keywords = pickle.load(fp)
    text1 = get_text(id)
    try:
        sentencesx1 = []
        sentencesx1 = text1.split(".")

        sentencesx = check(sentencesx1, keywords)
        text2 = " ".join(sentencesx)
        language = "en"
        max_ngram_size = 15
        deduplication_thresold = 0.8
        deduplication_algo = 'jaro'
        windowSize = 1
        numOfKeywords = 7
        all_keywords = []

        custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold,
                                                    dedupFunc=deduplication_algo, windowsSize=windowSize,
                                                    top=numOfKeywords,
                                                    features=None)
        keywordx = custom_kw_extractor.extract_keywords(text2)
        for kw in keywordx:
            x = str([match for match in sentencesx if str(kw[0]) in match]).replace("[", "").replace("]", "").replace(
                "'",
                "")
            all_keywords.append(str(kw[0]) + " | " + x + " |")
        y = ''.join(all_keywords)

        return Response(response=y, status=200, mimetype="application/json")
    except:
        return Response("No data found.", status=500, mimetype="application/json")


@app.route('/keywords_g/<int:id>', methods=['GET'])
def get_keywords_g(id):
    with open('keywords.pkl', 'rb') as fp:
        keywords = pickle.load(fp)
    text1 = get_text(id)

    sentencesx1 = []
    sentencesx1 = text1.split(".")

    sentencesx = check(sentencesx1, keywords)
    text2 = " ".join(sentencesx)

    model = KeyBERT('distilbert-base-nli-mean-tokens')
    all_keywords = []
    x1 = model.extract_keywords(text2, keyphrase_ngram_range=(5, 10), stop_words='english',
                                use_maxsum=True, nr_candidates=20, top_n=5)
    for kw in x1:
        all_keywords.append(str(kw) + '|')
    y = ''.join(all_keywords)
    resp = Response(response=y, status=200, mimetype='text/plain')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/summary/<int:id>', methods=['GET'])
def get_summa(id):
    with open('keywords.pkl', 'rb') as fp:
        keywords = pickle.load(fp)
    text1 = get_text(id, False)

    sentencesx1 = []
    sentencesx1 = text1.split(".")

    sentencesx = check(sentencesx1, keywords)
    text2 = ".".join(sentencesx)
    Albert_model = TransformerSummarizer(transformer_type="Albert", transformer_model_key="albert-base-v2")
    suma = ''.join(Albert_model(text2, min_length=60, num_sentences=8))
    resp = Response(response=suma, status=200, mimetype='text/plain')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp



@app.route('/summaryd/<int:id>', methods=['GET'])
def get_summaryd(id):
    with open('keywords.pkl', 'rb') as fp:
        keywords = pickle.load(fp)
    text1 = get_text_domain(id, False)
    sentencesx1 = []
    sentencesx1 = text1.split(".")
    sentencesx = check(sentencesx1, keywords)
    text2 = ".".join(sentencesx)
    Albert_model = TransformerSummarizer(transformer_type="Albert", transformer_model_key="albert-base-v2")
    suma = ''.join(Albert_model(text2, min_length=60, num_sentences=8))
    resp = Response(response=suma, status=200, mimetype='text/plain')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp




@app.route('/summary_I/<int:id>', methods=['GET'])
def get_summary_I(id):
    with open('keywords.pkl', 'rb') as fp:
        keywords = pickle.load(fp)
    text1 = get_text_impact(id, False)

    sentencesx1 = []
    sentencesx1 = text1.split(".")

    sentencesx = check(sentencesx1, keywords)
    text2 = ".".join(sentencesx)
    Albert_model = TransformerSummarizer(transformer_type="Albert", transformer_model_key="albert-base-v2")
    suma = ''.join(Albert_model(text2, min_length=60, num_sentences=8))
    resp = Response(response=suma, status=200, mimetype='text/plain')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp





@app.route('/summary_both', methods=['GET'])
def get_summary_both():
    id = request.args.get('id', None)
    id1 = request.args.get('id1', None)
    with open('keywords.pkl', 'rb') as fp:
        keywords = pickle.load(fp)
    text1 = get_text_both(id,id1, False)

    sentencesx1 = []
    sentencesx1 = text1.split(".")

    sentencesx = check(sentencesx1, keywords)
    text2 = ".".join(sentencesx)
    Albert_model = TransformerSummarizer(transformer_type="Albert", transformer_model_key="albert-base-v2")
    suma = ''.join(Albert_model(text2, min_length=60, num_sentences=8))
    resp = Response(response=suma, status=200, mimetype='text/plain')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp














@app.route('/qanda', methods=['GET'])
def get_qanda():
    id = request.args.get('id', None)
    question1 = request.args.get('quest', None)
    nlp_qa = pipeline('question-answering')
    model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
    nlp_qa = pipeline('question-answering')
    text1 = get_text(id, False)
    x = nlp_qa(context=text1, question=question1)
    y = "Answer:<b> " + x.get('answer') + "</b><br>"
    z = "Score :" + str(x.get('score')) + "</b><br>"
    xy = text1[x.get('start') - 250: x.get('start')] + "<b>" + text1[x.get('start'): x.get('end')] + '</b>' + text1[
                                                                                                              x.get(
                                                                                                                  'end'):x.get(
                                                                                                                  'end') + 250]
    resp = Response(y + z + " Context: " + xy, status=200, mimetype='text/HTML')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp



@app.route('/semsearch', methods=['GET'])
def get_semsearch():
    question1=request.args.get('quest',None)
    model = Word2Vec.load("Sem_search_model.model")
    df = pd.read_pickle("df_mesoc_abstract.pkl")
    q = spacy_tokenizer(question1)

    results = rank_docs(model, q, df, 10)
    xx=''
    for i in range(len(results)):
        if (results !=''):
            paper_name = results[i][0]

            sql1 = "Select ID_doc, author from paper where Title  like '%" + paper_name + "%'"

            c.execute(sql1)
            rows = c.fetchone()
            id = str(rows[0])
            author = rows[1]
            xx=xx+'<tr> <td><b> <b>'+str(i+1)+ '.   '+ author +':   ' + paper_name + '</td> <td></b>'
            xx=xx+'<a href="article.php?id='+id +'" "target="_self"><img src="images/view.png"  style="border: 0" alt="View"></td> </tr>'
        else:
            xx= "No data found"
    resp = Response(xx, status=200, mimetype='text/HTML')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/qa', methods=['GET'])
def get_qa():
    question2=request.args.get('quest',None)
    qa = text.SimpleQA(INDEXDIR)
    answers = qa.ask(question2)
    outx=""
    for a in answers[:10]:
        id =str(a['reference'])
        outx="<tr><td> Candidate answer : </td><td> <b>"+ a['answer'] +"</b></td><tr>"
        outx =outx +"<tr><td> Context: </td><td> <div>" + a['sentence_beginning'] + " <font color='red'>" + a['answer'] + "</font> " + a['sentence_end'] + "</div></td></tr>"
        outx= outx + "<tr><td> Confidence :</td><td>" + str(a['confidence']*100) +"% </td><tr>"
        outx= outx + "<tr><td> File:</td><td> <a href='article.php?id="+ id[:-4] + "' target='_self'><img src='images/view.png'  style='border: 0' alt='View'></td><tr>"
    resp =Response( outx , status=200, mimetype='text/HTML')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/gentext', methods=['GET'])
def get_gentext():
    text1=request.args.get('text',None)
    text_len=200
    device = torch.device('cpu')
    cur_ids = torch.tensor(tokenizer.encode(text1)).unsqueeze(0).long().to(device)

    model.eval()
    with torch.no_grad():
        for i in range(text_len):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0, -1],
                                           dim=0)  # Take the first(only one) batch and the last predicted embedding
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(),
                                            n=10)  # Randomly(from the given probability distribution) choose the next word from the top n words
            cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id],
                                dim=1)  # Add the last word

        output_list = list(cur_ids.squeeze().to('cpu').numpy())
        output_text = tokenizer.decode(output_list)
    resp = Response(output_text, status=200, mimetype='text/HTML')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
    



@app.route('/wordcloud/<int:id>', methods=['GET'])
def get_wordcloud(id):
    filename=str(id)+'wc.jpg'
    if  not os.path.exists(filename):
        c = conn.cursor()
        c.execute('SELECT abstract ||". "|| Body_text text FROM paper WHERE ID_Doc=' + str(id))
        x = str(c.fetchone())
        wc = WordCloud(background_color='white', max_words=100,  max_font_size=50,
                              random_state=42, width=400, height=200)
        wc.generate(x.lower())
        fig = plt.figure(1)
        #fig.set_size_inches(5, 5)
        plt.imshow(wc)
        plt.axis('off')
        plt.savefig(filename)
    return send_file(open(os.path.join(os.path.abspath(os.path.dirname(filename)), filename),'rb'), mimetype="image/jpeg",attachment_filename='"'+filename+'"')


@app.route('/candidate/<int:id>', methods=['GET'])
def get_candidate(id):
    c = conn.cursor()
    c.execute('SELECT abstract ||". "|| Body_text text FROM paper WHERE ID_Doc=' + str(id))
    text = str(c.fetchone())

    candidates = []
    stemmer = PorterStemmer()
    text = stemmer.stem(text)
    grammer = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+<VB.*>+(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>}'
    punct = set(string.punctuation)

    stop_words = set(nltk.corpus.stopwords.words('english'))
    chunker = nltk.chunk.regexp.RegexpParser(grammer)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))

    for i in range(len(tagged_sents)):
        all_chunks = nltk.chunk.tree2conlltags(chunker.parse(tagged_sents[i]))
        c = itertools.groupby(all_chunks, key=lambda x: x[2])
        candid = [' '.join(x[0] for x in group) for key, group in itertools.groupby(all_chunks, lambda x: x[2] != 'O')
                  if key]
        candidates = candidates + candid
    candidates = list(set(candidates))
    candidates = [word for word in candidates if len(word.split()) > 3]

    sql2 = "SELECT Keywords  FROM keywords "
    keywords = pd.read_sql_query(sql2, conn)
    keywords_a = keywords["Keywords"].to_list()
    keywords_a = [element.lower() for element in keywords_a]

    strx=""
    for i in candidates:
        if any(keywords_a in i for keywords_a in keywords_a):
            strx=strx+"<tr><td>" +  i + "</tr></td>"
    resp = Response(strx , status=200, mimetype='text/HTML')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return(resp)


@app.route('/class_w/<int:id>', methods=['GET'])
def get_class_w(id):
    c = conn.cursor()
    c.execute('SELECT abstract  FROM paper WHERE ID_Doc=' + str(id))
    text = str(c.fetchone())
    predictor = ktrain.load_predictor('e:/pycharm_projects/mesoc_api/W_classify')
    strw=str(predictor.predict(text))
    resp = Response(strw, status=200, mimetype='text/HTML')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return (resp)



@app.route('/class_d/<int:id>', methods=['GET'])
def get_class_d(id):
    c = conn.cursor()
    c.execute('SELECT abstract  FROM paper WHERE ID_Doc=' + str(id))
    text = str(c.fetchone())
    predictor = ktrain.load_predictor('e:/pycharm_projects/mesoc_api/DOM_classify')
    strw=str(predictor.predict(text))
    resp = Response(strw, status=200, mimetype='text/HTML')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return (resp)


@app.route('/qanda_D', methods=['GET'])
def get_qanda_D():
    id = request.args.get('id', None)
    question1 = request.args.get('quest', None)
    nlp_qa = pipeline('question-answering')
    model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
    nlp_qa = pipeline('question-answering')
    text1 = get_text_domain(id, False)
    x = nlp_qa(context=text1, question=question1)
    y = "Answer:<b> " + x.get('answer') + "</b><br>"
    z = "Score :" + str(x.get('score')) + "</b><br>"
    xy = text1[x.get('start') - 250: x.get('start')] + "<b>" + text1[x.get('start'): x.get('end')] + '</b>' + text1[
                                                                                                              x.get(
                                                                                                                  'end'):x.get(
                                                                                                                  'end') + 250]
    resp = Response(y + z + " Context: " + xy, status=200, mimetype='text/HTML')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp



@app.route('/qanda_I', methods=['GET'])
def get_qanda_I():
    id = request.args.get('id', None)
    question1 = request.args.get('quest', None)
    nlp_qa = pipeline('question-answering')
    model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
    nlp_qa = pipeline('question-answering')
    text1 = get_text_impact(id, False)
    x = nlp_qa(context=text1, question=question1)
    y = "Answer:<b> " + x.get('answer') + "</b><br>"
    z = "Score :" + str(x.get('score')) + "</b><br>"
    xy = text1[x.get('start') - 250: x.get('start')] + "<b>" + text1[x.get('start'): x.get('end')] + '</b>' + text1[
                                                                                                              x.get(
                                                                                                                  'end'):x.get(
                                                                                                                  'end') + 250]
    resp = Response(y + z + " Context: " + xy, status=200, mimetype='text/HTML')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp



@app.route('/qanda_both', methods=['GET'])
def get_qanda_both():
    id = request.args.get('id', None)
    id1 = request.args.get('id1', None)
    question1 = request.args.get('quest', None)
    nlp_qa = pipeline('question-answering')
    model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
    nlp_qa = pipeline('question-answering')
    text1 = get_text_both(id,id1, False)
    x = nlp_qa(context=text1, question=question1)
    y = "Answer:<b> " + x.get('answer') + "</b><br>"
    z = "Score :" + str(x.get('score')) + "</b><br>"
    xy = text1[x.get('start') - 250: x.get('start')] + "<b>" + text1[x.get('start'): x.get('end')] + '</b>' + text1[
                                                                                                              x.get(
                                                                                                                  'end'):x.get(
                                                                                                                  'end') + 250]
    resp = Response(y + z + " Context: " + xy, status=200, mimetype='text/HTML')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp