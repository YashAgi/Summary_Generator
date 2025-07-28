import streamlit as st
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
import re

st.sidebar.subheader("The Top Sentences are : ")
lines = st.sidebar.slider("Select Number Of Lines : ",min_value=3,max_value=10,value=5)

st.title("Welcome, This is Summary Generator App")

corpus = st.text_input("Enter the text here : ")

## loading model aganin and again takes a lot time, so, load it only once and after that use its instance
@st.cache_resource
def load_model():
    return api.load('word2vec-google-news-300')

wv = load_model()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def getSummary(text):
    #firstly convert the text into sentences
    corpus = nltk.sent_tokenize(text)

    processed_corpus = []
    filtered_tokens = []
    vectors = []

    avg_corpus_vector = np.zeros(300) # create an array of 300 zeroes(empty vectore)

    # remove all the stopwords from the corpus
    for i in range(len(corpus)):
        sentence = re.sub(r'[^\w\s]', '', corpus[i])  # Remove punctuation
        words = nltk.word_tokenize(sentence)
        
        words = [word for word in words if word.lower() not in set(stopwords.words('english')) and word.lower() in wv]
        
        if len(words) == 0:
            continue  # Skip empty sentences after filtering
    
        processed_corpus.append(corpus[i])     # Keep the original cleaned sentence
        filtered_tokens.append(words)          # Save the tokenized filtered list for vectorization

    #Vectorization
    for i in range(len(filtered_tokens)):
        vector_sum = np.zeros(300)
        for word in filtered_tokens[i]:
            vector_sum += wv[word]
        
        sentence_vector = vector_sum / len(filtered_tokens[i])
        vectors.append((processed_corpus[i], sentence_vector))
        avg_corpus_vector += sentence_vector

        
    avg_corpus_vector = avg_corpus_vector/len(processed_corpus)

    sorted_sentences = sorted(vectors, key=lambda x: cosine_similarity(x[1], avg_corpus_vector), reverse=True)
    top_N_sentences = [x[0] for x in sorted_sentences[:lines]]

    return top_N_sentences, corpus

if st.button("Generate Summary"):
    summary,corpus = getSummary(corpus)

    st.markdown("### 🔍 Full Text with Highlighted Summary Sentences:")

    # Join all sentences into one paragraph, styling the summary ones, highlight the important sentences with orange color.
    highlighted_text = " ".join([
    f"<span style='background-color: orange; font-weight: bold'>{sentence}</span>" if sentence in summary else sentence
    for sentence in corpus
    ])
    st.markdown(highlighted_text, unsafe_allow_html=True)

    for i, sentence in enumerate(summary):
        st.sidebar.write(f"{i+1}. {sentence}")
