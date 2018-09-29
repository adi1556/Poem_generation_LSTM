import sys
import numpy as np
import pandas as pd
import itertools
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import random
from random import choice

def split_data(address):
    """ 
        Reads in a text of poems with poems and titess and makes a separate 
        list of poems and titles.Poems from Project Gutenberg were usually in 
        this format. Please adjust line breaks accordingly if different.
    """
    
    poem = []
    title =[]
    text=(open(address).read())
    new_text = text.split('\n\n\n\n\n')
    poet_series = pd.Series(new_text)
    line_split = poet_series.apply(lambda x: x.split('\n\n\n'))
    for i in range(len(line_split)-3):
        poem.append(line_split[i][1])
        title.append(line_split[i][0])
    return(poem, title) 


def display_topics(model, feature_names, num_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-num_top_words - 1:-1]]))


def topic_modeling(gen, n_components):
    """
        Takes in a cursor generator and number of componenets for LDA and returns topics
    """
    count_vectorizer = CountVectorizer(ngram_range=(1, 2),  
                                   stop_words='english', 
                                   token_pattern="\\b[a-z][a-z]+\\b",
                                   lowercase=True,
                                   max_df = 0.6)

    count_vectorizer.fit(gen)
    lda = LatentDirichletAllocation(n_components)
    for _ in range(10):
    
        for file in gen:
            vec_file = count_vectorizer.transform([file])
            lda.partial_fit(vec_file)
    return(display_topics(lda, count_vectorizer.get_feature_names(), 10))


def pattern_and_label(gen,seq_length):
    '''
        Breaks down the poem in a pattern of specified sequence length and the 
        next character after the patter is termed as label. 
        
        Input
        -----------------------------------------------------------------------
        A cursor generator to go through every poem: gen
        The length of sequence/pattern which needs to be learnt: seq_length
        
        Output
        -----------------------------------------------------------------------
        Normalized floats for every character in poems: X_modified
        Normalized character for every label of certain pattern of characters:
        Y_modified
        All characters in the poem as number: X 
        List of all characters in the poems: characters
    '''
    poems_list = list(x['poems'] for x in gen)
    characters = sorted(set(list(itertools.chain(*poems_list))))
    
    char_to_n = {char:n for n,char in enumerate(characters)}

    
    for x in poems_list:
         length = len(list(itertools.chain(*x)))   
         raw_text = list(itertools.chain(*x))
         
         X = []
         Y = []

         for i in range(0, length-seq_length, 1):
            sequence = raw_text[i:i + seq_length]
            label = raw_text[i + seq_length]
            X.append([char_to_n[char] for char in sequence])
            Y.append(char_to_n[label])

            X_modified = np.reshape(X, (len(X), seq_length, 1))
            X_modified = X_modified / float(len(characters))
            Y_modified = np_utils.to_categorical(Y)
    return (X_modified, Y_modified, X, characters)
    
    
    
def model_generation(X_modified, Y_modified, size):    
    '''
        Generates an LSTM model with specified size and 3 layers. I used size 700
        
        Input
        -----------------------------------------------------------------------
        Normalized floats for every character in poems: X_modified
        Normalized character for every label of certain pattern of characters:
        Y_modified
        Size of every layer: size
        
        Output
        -----------------------------------------------------------------------
        Model with specified size and 3 layers: model
    '''
    model = Sequential()
    model.add(LSTM(size, input_shape=(X_modified.shape[1], X_modified.shape[2]), 
                   return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(size, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(size))
    model.add(Dropout(0.2))
    model.add(Dense(Y_modified.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model


def sample(preds, temperature=1.0):
    ''' 
        Helper function to sample an index from a probability array
    
    '''
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def model_fit(model, X_modified, Y_modified, epochs):
    """
        Takes in a defined model and trains of all the X and Y values. X being 
        the pattern of characters and Y being the label for those characters.
        
        Input
        -----------------------------------------------------------------------
        Normalized floats for every character in poems: X_modified
        Normalized character for every label of certain pattern of characters:
        Y_modified
        An architecture of model to fit on: model
        Number of epochs for training: epochs
        
        
        Output
        -----------------------------------------------------------------------
        Fitted model on all poems -- model

    """      
    model.fit(X_modified, Y_modified, epochs=epochs, batch_size=100)
            
    return model

    
def text_generator(X, model, characters, num_characters):  
    '''
        Generates text after looking at a random string of characters.
        
        Input
        -----------------------------------------------------------------------
        Fitted model on all poems -- model
        Numbers for every character in sequence --X
        All characters in the poems -- characters
        Characters to be predicted: num_characters
        
        Output
        -----------------------------------------------------------------------
        String of characters
    
    '''
    
    string_mapped = random.choice(X)
    n_to_char = {n:char for n,char in enumerate(characters)}
    for diversity in [0.1, 0.2, 0.3]:
        for i in range(num_characters):
            x = np.reshape(string_mapped, (1, len(string_mapped), 1))
            x = x / float(len(characters))
            prediction = model.predict(x, verbose=0)[0]
            index = sample(prediction, diversity)
            result = n_to_char[index]
            sys.stdout.write(result)
            
            string_mapped.append(index)
            string_mapped = string_mapped[1:len(string_mapped)]
        print('\n')
        