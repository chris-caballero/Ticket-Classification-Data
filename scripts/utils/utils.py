import numpy as np
import pandas as pd

def get_data(filename):
    """
    Load data from a JSON file and convert it into a Pandas DataFrame.

    Args:
        filename (str): Name of the JSON file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    import json

    print('Loading data...')
    with open(filename) as f:
        data = json.load(f)
        df = pd.json_normalize(data)
        df.columns = ['index', 'type', 'id', 'score', 'tags', 'zip_code', 'complaint_id', 'issue', 'date_received',
                      'state', 'consumer_disputed', 'product', 'company_response', 'company', 'submitted_via',
                      'date_sent_to_company', 'company_public_response', 'sub_product', 'timely',
                      'complaint', 'sub_issue', 'consent_provided']
        df[df['complaint'] == ''] = np.nan
        df.dropna(subset=['complaint'], inplace=True)
    
    return df

def clean(text):
    """
    Clean and preprocess text by converting to lowercase, removing non-alphanumeric characters,
    digits, and common placeholders.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    import re

    text = text.lower()
    text = re.sub('[^\w\s]', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = text.replace('xxxx', '')
    text = text.strip()
    return text

def lemmatize(text):
    """
    Lemmatize text using spaCy.

    Args:
        text (str): Input text.

    Returns:
        str: Lemmatized text.
    """
    import en_core_web_sm
    nlp = en_core_web_sm.load()
    doc = nlp(text)
    return ' '.join(tok.lemma_ for tok in doc)

def noun_extraction(text):
    """
    Extract nouns from text using TextBlob.

    Args:
        text (str): Input text.

    Returns:
        str: Extracted nouns.
    """
    from textblob import TextBlob
    blob = TextBlob(text)
    return ' '.join(tok for (tok, tag) in blob.tags if tag == 'NN')

def preprocess(df, fname=None):
    """
    Preprocess the complaints data.

    Args:
        df (pd.DataFrame): Original DataFrame containing complaints data.
        fname (str, optional): File name to save preprocessed data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame containing cleaned, lemmatized, and noun-extracted text.
    """
    print('Cleaning Text...')
    complaints = pd.DataFrame(df['complaint'].apply(clean))
    print('Sample:', complaints['complaint'].sample(1))
    
    print('\nLemmatizing Text...')
    complaints['complaint_lemma'] = complaints['complaint'].apply(lemmatize)
    print('Sample:', complaints['complaint_lemma'].sample(1))
    
    print('\nExtracting POS (NN)...')
    complaints['complaint_nouns'] = complaints['complaint_lemma'].apply(noun_extraction)
    print('Sample:', complaints['complaint_nouns'].sample(1))
    
    if fname is not None:
        print('Saving...')
        complaints.to_pickle(fname)
    
    print('Done!')
    return df, complaints

def word_frequencies(text, pipeline=None, n=1):
    """
    Calculate word frequencies in the text.

    Args:
        text (pd.Series): Series of text data.
        pipeline (sklearn.pipeline.Pipeline, optional): Text processing pipeline.
        n (int, optional): N-gram range for word frequencies.

    Returns:
        list: List of tuples containing word-frequency pairs.
        scipy.sparse.csr_matrix: Document-term matrix.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    if pipeline is None:
        vectorizer = CountVectorizer(ngram_range=(n, n))
        X = vectorizer.fit_transform(text)
    else:
        vectorizer = pipeline['count']
        X = vectorizer.transform(text)
        
    frequency = X.sum(axis=0)
    word_frequency = [(word, frequency[0, vectorizer.vocabulary_[word]]) for word in vectorizer.vocabulary_]
    word_frequency = sorted(word_frequency, key=lambda x: x[1], reverse=True)
    
    return word_frequency, X

def show_top_words(text=None, pipeline=None, word_frequency=None, num_samples=10):
    """
    Display the top words by frequency in a bar plot.

    Args:
        text (pd.Series, optional): Series of text data.
        pipeline (sklearn.pipeline.Pipeline, optional): Text processing pipeline.
        word_frequency (list, optional): List of tuples containing word-frequency pairs.
        num_samples (int, optional): Number of top words to display.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    if word_frequency is None:
        if text is None:
            print('Please provide the corpus or a word-frequency mapping!')
            return
        word_frequency, _ = word_frequencies(text, pipeline=pipeline, n=1)
        
    plt.figure(figsize=[10, 4])
    top_words = word_frequency[:num_samples]
    items = [x[0] for x in top_words]
    counts = [x[1] for x in top_words]
    
    plt.bar(x=items, height=counts, color=sns.color_palette('pastel'))

    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.title(f"Top {num_samples} words by frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def vectorizer_pipeline(text):
    """
    Create a text vectorization pipeline.

    Args:
        text (pd.Series): Series of text data.

    Returns:
        sklearn.pipeline.Pipeline: Text vectorization pipeline.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.decomposition import NMF

    vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    nmf = NMF(n_components=5)
    pipe = Pipeline([('count', vectorizer), ('tfidf', tfidf_transformer), ('nmf', nmf)])
    pipe = pipe.fit(text)
    
    return pipe
