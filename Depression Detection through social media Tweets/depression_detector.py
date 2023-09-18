# ===============================================================
# Import necessary libraries
# ===============================================================
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def detect_depression(text):
    '''
    text is a list of string.
    returns: whether the person is normal or is in depression
    '''

    # Load the mood detector
    mood_detector = tf.keras.models.load_model("mood_detector")

    # Load the tokenizer from the saved file
    with open('tokenizer.pkl', 'rb') as token_file:
        loaded_tokenizer = pickle.load(token_file)

    # Tokenize new text data using the loaded tokenizer
    text_seq = loaded_tokenizer.texts_to_sequences(text)
    text_pad = pad_sequences(
        text_seq, maxlen=50, padding='post', truncating='post')

    # Predict
    predictions = mood_detector.predict(text_pad)

    ''' If hate probability is greater than 80% and offensive probability is 
  greater than 70%, only then we'll declare that the person who posted the
  content is in state of depression.'''

    if predictions[0][0] > 0.8 and predictions[0][1] > 0.7:
        print("This person is in Depression!")
    else:
        print("This person is in Normal state!")


# ===============================================================
# TESTING THE MODEL - Change the string within 'text list'!
# ===============================================================
text = ["I love this food soooo much oh my god :)"]
detect_depression(text)
