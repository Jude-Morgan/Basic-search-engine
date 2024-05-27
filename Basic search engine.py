import nltk

nltk.download('stopwords')
nltk.download('punkt')

import re
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import nltk

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('punkt')

# Initialize stemmer and stop-words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def tokenize(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Split text into words
    tokens = nltk.word_tokenize(text)
    # Remove stop-words and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(lambda: defaultdict(int))
        self.doc_freq = defaultdict(int)
        self.doc_lengths = defaultdict(int)
        self.num_docs = 0
    
    def add_document(self, doc_id, text):
        self.num_docs += 1
        tokens = tokenize(text)
        unique_tokens = set(tokens)
        
        for token in unique_tokens:
            self.doc_freq[token] += 1
        
        for token in tokens:
            self.index[token][doc_id] += 1
            self.doc_lengths[doc_id] += 1
    
    def _tf_idf(self, term, doc_id):
        tf = self.index[term][doc_id] / self.doc_lengths[doc_id]
        idf = math.log(self.num_docs / (1 + self.doc_freq[term]))
        return tf * idf
    
    def _phrase_search(self, phrase, doc_id, text):
        phrase_tokens = phrase.split()
        text_tokens = text.split()
        text_len = len(text_tokens)
        phrase_len = len(phrase_tokens)
        
        for i in range(text_len - phrase_len + 1):
            if text_tokens[i:i + phrase_len] == phrase_tokens:
                return True
        return False

    def search(self, query, phrase_search=False):
        if phrase_search:
            matching_docs = []
            for doc_id, text in documents.items():
                if self._phrase_search(query, doc_id, text):
                    matching_docs.append(doc_id)
            return matching_docs
        else:
            tokens = tokenize(query)
            if not tokens:
                return []
            
            scores = defaultdict(float)
            
            for token in tokens:
                if token in self.index:
                    for doc_id in self.index[token]:
                        scores[doc_id] += self._tf_idf(token, doc_id)
            
            ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [doc_id for doc_id, score in ranked_results]
        

def process_query(inverted_index, query, phrase_search=False):
    results = inverted_index.search(query, phrase_search=phrase_search)
    return results


# Sample documents
documents = {
    1: "The quick brown fox jumps over the lazy dog",
    2: "Never jump over the lazy dog quickly",
    3: "A quick brown dog outpaces a fast fox",
    4: "Artificial intelligence and machine learning are transforming industries",
    5: "Deep learning is a subset of machine learning that deals with neural networks",
    6: "The history of the world is a story of continuous evolution and adaptation",
    7: "The theory of relativity fundamentally changed our understanding of physics",
    8: "Quantum computing represents a significant leap forward in computational power",
    9: "Climate change is a pressing global issue that requires immediate action",
    10: "The development of renewable energy sources is crucial for a sustainable future",
}


# Initialize inverted index
index = InvertedIndex()

# Add docs to index
for doc_id, text in documents.items():
    index.add_document(doc_id, text)

# Process queries
queries = [
    ("machine learning", False),
    ("quick brown fox", False),
    ("climate change", False),
    ("quantum computing", False),
    ("artificial intelligence", True),
    ("deep learning neural networks", True)
]

# Search and print results
for query, phrase_search in queries:
    results = process_query(index, query, phrase_search=phrase_search)
    print(f"Query: '{query}' (Phrase search: {phrase_search}) -> Results: {results}")

