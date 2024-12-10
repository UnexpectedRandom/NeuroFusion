from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

config = {
    'max_words':900
}

ntoken = Tokenizer(num_words = config['max_words'])

def TokenizerText(text):
    ntoken.fit_on_texts(text)
    list_words = text_to_word_sequence(text)
    
    return list_words