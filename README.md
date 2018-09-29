# Poem_generation_LSTM

This project was aimed at understanding LSTM's and how they are used. I used LSTM's to generate poetry. I collected poems from Project Gutenberg and trained my model using that on a GPU instance of AWS for 8 hours. 

The model was able to learn spellings, punctuations, when to change lines and some rhyming scheme as well.

func_text_gen.py has all the functions for cleaning the text, pattern and label creation from poems, model creation as well as text generation. 

Poem_generator.ipynb has used these functions to show how to use them. 

Apart from poetry generation, I also wanted to see if I could do some topic modeling. I didn't have a lot of data so topic modeling was inconsequential but the files still have the function if you want to do that with lots of data.

Finally I tried VADER Sentiment Analyzer to check the sentiment of the poems. 

A line my generator generated:
### I see the world when I see your eyes

If that doesn't get you brownie points from your spouse, I'm not sure what else will. 
