__author__ = 'oren'
import re
import word2vec

def clean_text(text):
    text = text.lower()
    text = re.sub('\'\"', ' ', text)
    text = re.sub('\n|,|[^A-Za-z]', ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.strip()
    return text


if __name__ == '__main__':
    w2v_model = word2vec.load('text8.bin')
    words_dict = dict([(vv[0], (i, vv[1])) for i, vv in enumerate(zip(w2v_model.vocab, w2v_model.vectors))])
    all_sentences = get_sentences(TEXT_DATA)
    sentences, sentences_embed, sentences_w2v = get_sentences(all_sentences, words_dict)

    export_sentences(sentences_embed, '/home/orent/models/word2vec/sentences_embed')
    export_w2v(w2v_model,
               '/home/orent/models/word2vec/w2v_mat',
               '/home/orent/models/word2vec/w2v_words')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              