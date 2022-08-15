# MultiSeq2seq transformer model for NLP
As a general rule, increasing the number of relevant features is a good way to improve an ML model accuracy. In case of seq2seq models you can implement this concept by using additional input sequences, also known as covariates. 

## **NER tagging task**

As an example of using a seq2seq model that can take multiple input sequences, consider using a transformer for the task of NER. Such a model could take the following sequences:

- the sequence of tokens that make up the input text (many models use only this one)
- the sequence of the corresponding fine-grained part-of-speech (POS) tags
- the sequence of the words that are most syntactically related to the words that can potentially be NER.

Throughout the rest of this tutorial, we discuss the implementation in Python. 
```python
import numpy as np
import pandas as pd
```
We'll use the NER dataset available from Kaggle at https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus. So, you'll need to download and unpack it, in which we'll need ner_dataset.csv. Below, we load the data into a pandas DataFrame to simplify upcoming preprocessing tasks.
```python
df = pd.read_csv('ner_dataset.csv', encoding="latin1")
df = df.fillna(method = 'ffill')
df.head()
```
Let's now get the list of words found in the corpus, and add one more "PAD" for padding purposes.
```python
words = list(df['Word'].unique())
words.append("[PAD]")
num_words = len(words)
print("Number of unique words: ", num_words)
```
Another input sequence for our model will include POS tags. So we need to create a list of POS tags used in the corpus.
```python
POS = list(df['POS'].unique())
POS.append("[X]")
print(POS)
num_POS = len(POS)
```
We also need to get the list of NER tags to be used as prediction classes.
```python
tags = list(df['Tag'].unique())
print(tags)
```
Named entities can appear in a chunks of word, say: 'New York City'. We're using BIO tagging to identify boundaries:

- B-{entity} : Mark the begining of an entity chunk.
- I-{entity} : Mark word inside the chunk.
- O : Mark word outside the chunk.

Suppose that in this particular assignment, we need only ['O', 'B-geo', 'I-geo'] tags, while all the others can be removed (just replaced with 'O' in the DataFrame).
```python
df['Tag'] = df['Tag'].replace(['B-gpe', 'B-gpe', 'B-per', 'B-org', 'I-org', 'B-tim', 'B-art', 'I-art', 'I-per', 'I-gpe', 'I-tim', 'B-nat', 'B-eve', 'I-eve', 'I-nat'], 'O')

tags = list(df['Tag'].unique())
print(tags)
num_tags = len(tags)
```
In the next step, we group the data by sentence, including only the word, POS  and tag fields in the resuling dataset. 
```python
df['Sentence #'] = df['Sentence #'].apply(lambda x: x[10:])
df['Sentence #'] = pd.to_numeric(df['Sentence #'])
data_grouped = df.groupby("Sentence #").apply(lambda x:[(word,tag,pos) for word, tag, pos in zip(x["Word"].values.tolist(),  x["Tag"].values.tolist(), x["POS"].values.tolist())])
data_grouped.head(10)
```
For convenience, we convert the Series into a list:
```python
sent_groups = [sent_group for sent_group in data_grouped]
```
Now that we've grouped the data by sentence, it's important to determine the max length of such a group, thus determining the max sequence length to be used in further processing. 
```python
seqs_length = [len(s) for s in sent_groups]
max_seq_length = max(seqs_length)
print("Max sequence length: ", max_seq_length)
```
Thinking about improving the model accuracy, it's always a good idea to consider using more features. In this particular assigment, you might consider another input sequence to be used along with the sequences of tokens and their POS tags. 

This can be, for example, a sequence in which you use the syntactic head of a token if it is a direct object, the head of head for an object of preposition, and it's going to be a fill symbol for all the other words in a sentence. 

This new sequence is intended to give the model information about the words that are syntactically related to those ones that can be potentially NER. For example, in the utterance "I like autumn in Paris.", word "autumn" will be associated with word "Paris", since "autumn" is the head of head of "Paris" and the latter is an object of preposition. 

To implement this, we first need to do some additional preprocessing of the dataset. In particular, we need to recreate sentences from the words found in the Word column of the dataset, so that we can then do syntactic dependency analysis against those sentences.
```python
sentences = df.groupby("Sentence #").apply(lambda x:(' '.join(x["Word"].values.tolist()))) 
```
The following simple test shows that the number of sentences is equal to the number of sent groups calculated previously.  
```python
len(list(sentences))

len(sent_groups)
```
Now we are ready to do syntactic dependency analysis with tools like spaCy.
```python
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
nlp = spacy.load('en_core_web_sm', disable = ['ner'])
```
In our dataset, words and numbers that contain hyphens come through as a single token. That is not the default spaCy's tokenizer behavior however. So we need to take care of it by creating a custom tokenizer.
```python
def custom_tokenizer(nlp):
    inf = list(nlp.Defaults.infixes)               # Default infixes
    inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")    # Remove the generic op between numbers or between a number and a -
    inf = tuple(inf)                               # Convert inf to tuple
    infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])  # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
    infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x] # Remove - between letters rule
    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)

nlp.tokenizer = custom_tokenizer(nlp)
```
Just a simple test to make sure it works as expected:
```python
doc = nlp('The London march came ahead of anti-war protests today in other cities , including Rome , Paris , and Madrid .')
for t in doc:
  print(t.text, t.pos_)
```
Now, we can proceed to our syntactic dependency analysis:
```python
heads = []
for sent in list(sentences):
  l=[]
  doc = nlp(sent)
  for t in doc:
    if (t.dep_ == 'dobj'):
      l.append(t.head.text)
    elif (t.dep_ == 'pobj'):
      l.append(t.head.head.text)
    else:
      l.append('[PAD]')
  heads.append(l)
```

The total number of elements in the sequence we just got must be equal to the number of elements in the other sequences.
```python
len(heads)
```
Now we can move on to the next step of input data preproccesing and convert the text data into numbers. The point is, our model will expect numbers for processing, rather than words. So we need to do the following conversion, assigning each word in the vocabulary to a unique integer - the same for POS and NER tags. 
```python
word2id = {word: id for id, word in enumerate(words)}
tag2id = {tag: id for id, tag in enumerate(tags)}
pos2id = {pos: id for id, pos in enumerate(POS)}
```
Each word in the text can be converted now to its corresponding number - as well as the tags.  
```python
X1 = [[word2id[w[0]] for w in s] for s in sent_groups]
X2 = [[pos2id[w[2]] for w in s] for s in sent_groups]
X3 = [[word2id[w] for w in s] for s in heads]
y = [[tag2id[w[1]] for w in s] for s in sent_groups]
```
Finally, we need to pad our sentences (actually number sequences) to the same length.
```python
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
X1= pad_sequences(maxlen = max_seq_length,sequences = X1,padding = 'post',value = num_words-1)
X2= pad_sequences(maxlen = max_seq_length,sequences = X2,padding = 'post',value = num_POS-1)
X3= pad_sequences(maxlen = max_seq_length,sequences = X3,padding = 'post',value = num_words-1)
Y = pad_sequences(maxlen = max_seq_length,sequences = y,padding = 'post',value = tag2id['O'])
```
Now we can combine the vectors to be used as the input sequences into a single matrix to be used as a unified input of the model.
```python
X =np.stack((X1,X2,X3), axis = 1)
X = X.reshape(X1.shape[0], 3, X1.shape[1])
```
Then, we convert the y vector to a binary class matrix.
```python
y = np.array([to_categorical(i, num_classes = num_tags) for i in Y], dtype='int')
```
Now that we have X and y to be used in training and evaluating the model, it would be interesting to look at their shapes.
```python
print('X shape', X.shape, 'y shape', y.shape)
```
The next step is to split the X and Y sets into training and test sets.
```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state=1)
```
Now we can define a model to be trained on the above data. Below is an implementation of the typical transformer architecture adjusted to the task in hand. This arcitecture will be used as the prediction model for this assignment.
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
#the embedding block implemented in a separate class 
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, max_pos, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=max_pos, output_dim=embed_dim)
        self.position_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.position_emb(positions)
        x0=tf.reshape(x[:,0:1,0:maxlen], [-1,maxlen])
        x0 = self.token_emb(x0)
        x1=tf.reshape(x[:,1:2,0:maxlen], [-1,maxlen])
        x1 = self.pos_emb(x1)           
        x2=tf.reshape(x[:,2:3,0:maxlen], [-1,maxlen])
        x2 = self.token_emb(x2)
        x = tf.stack([x0,x1,x2], axis=1)
        return x + positions
```
Before proceeding to assembling the model, we need to define its parameters.
```python
vocab_size = num_words # vocabulary size
max_pos = num_POS # the number of POS tags
sequence_length = max_seq_length # the max length of a sequence
batch_size = 32 # batch size
embed_dim = 128  # Embedding size for each token
num_heads = 8  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer
```
Here is the model assembly.
```python
inputs = layers.Input(shape=(3, sequence_length, ), dtype="int64")
embedding_layer = TokenAndPositionEmbedding(sequence_length, max_pos, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.AveragePooling2D(pool_size=(3, 1), data_format='channels_last')(x)
x = tf.keras.layers.Reshape((sequence_length, embed_dim))(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(200, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(3, activation="softmax")(x)
#creating the model
model = keras.Model(inputs=inputs, outputs=outputs)
```
Compiling and training the model.
```python
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print(model.summary())
history = model.fit(x_train, y_train, validation_split =0.2, batch_size=batch_size, epochs=1)
```
Now we can evaluate the model on the test data.
```python
model.evaluate(x_test,y_test)
```
Let's test our model on a new (previuosly unseen) sentence, as well as a previously unseen geo-NER. First try the first sentence, then the second. You'll see that the prediction depends a lot on the context (words surrounding the word in question).
```python
new = "I worked for Rongo."
new = "I am going to Rongo."
#Then, you can try these sentences to see how it works with NERs that are obviously not geo-NERs.
#new = "I worked for Tom."
#new = "I am going to Tom."
X1_new=[]
X2_new =[]
X3_new=[]
doc = nlp(new)
for t in doc:
  try:
    X1_new.append(word2id[t.text])
  except:
    X1_new.append(word2id["[PAD]"])
  X2_new.append(pos2id[t.tag_])
  try:
    if (t.dep_ == 'dobj'):
      X3_new.append(word2id[t.head.text])
    elif (t.dep_ == 'pobj'):
      X3_new.append(word2id[t.head.head.text])
    else:
      X3_new.append(word2id['[PAD]'])
  except:
    X3_new.append(word2id['[PAD]'])
X1_new = keras.preprocessing.sequence.pad_sequences(sequences=[X1_new], maxlen=max_seq_length, padding='post', value = num_words-1)
X2_new = keras.preprocessing.sequence.pad_sequences(sequences=[X2_new], maxlen=max_seq_length, padding='post', value = num_POS-1)
X3_new = keras.preprocessing.sequence.pad_sequences(sequences=[X3_new], maxlen=max_seq_length, padding='post', value = num_words-1)
X_new =np.stack((X1_new,X2_new,X3_new), axis = 1)
predictions = model.predict(X_new)
```
You can tune the threshold as needed.
```python
threshold = 0.9
id2tag = {id: tag for id, tag in enumerate(tags)}
for i in range(len(doc)):
    print(doc[i].text, id2tag[predictions.argmax(axis=-1)[0][i]]) 
    if predictions[0][i][0]<threshold:
      print(predictions[0][i]) 
```
Here is what the result might look like:
```python
I O
am O
going O
to O
Rongo B-geo
[0.34629568 0.6167148  0.03698955]
. O
```
