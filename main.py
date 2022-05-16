import pickle
import tensorflow as tf
import numpy as np
import csv
import keras
from keras import layers
from os import path
import re

# Defining hyperparameters

VOCAB_SIZE = 8192*2
MAX_SAMPLES = 50000*2
BUFFER_SIZE = 20000*2
MAX_LENGTH = 40*2
EMBED_DIM = 256*2
LATENT_DIM = 512*2
NUM_HEADS = 8*2
BATCH_SIZE = 64

# path_to_zip = tf.keras.utils.get_file(
#     "data/cornell_movie_dialog.zip",
#     origin="http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
#     extract=True)

# path_to_dataset = path.join(path.dirname(path_to_zip), "data/cornell movie-dialogs corpus")
path_to_dataset = "data/cornell movie-dialogs corpus"
path_to_movie_lines = path.join(path_to_dataset, "movie_lines.txt")
path_to_movie_conversations = path.join(path_to_dataset, "movie_conversations.txt")

def load_conversations():
   """Helper function for loading the conversation splits"""
   id2line = {}
   with open(path_to_movie_lines, errors="ignore") as file:
      lines = file.readlines()

   for line in lines:
      parts = line.replace("\n", "").split(" +++$+++ ")
      id2line[parts[0]] = parts[4]
       
   inputs, outputs = [], []

   with open(path_to_movie_conversations, "r") as file:
      lines = file.readlines()
      
   for line in lines:
      parts = line.replace("\n", "").split(" +++$+++ ")
      # get conversation in a list of line ID
      conversation = [line[1:-1] for line in parts[3][1:-1].split(", ")]
      for i in range(len(conversation) - 1):
         inputs.append(id2line[conversation[i]])
         outputs.append(id2line[conversation[i + 1]])
         if len(inputs) >= MAX_SAMPLES:
            return inputs, outputs
   return inputs, outputs

def read_Trump(path_to_csv_file):
   with open(path_to_csv_file, 'r', errors="ignore") as file:
      csvreader = csv.reader(file)
      _ = next(csvreader) # Header

      questions, trump_line = [], []
      
      rows = []
      for row in csvreader:
         if "Trump" in str(row[0]):
            questions.append(rows[-1][2])
            trump_line.append(row[2])
               
         rows.append(row)
            
   return questions, trump_line
 
questions, answers = load_conversations()
q2, a2 = read_Trump("data/us_election_2020_2nd_presidential_debate.csv")
q3, a3 = read_Trump("data/us_election_2020_1st_presidential_debate.csv")
q1, a1 = read_Trump("data/us_election_2020_trump_town_hall.csv")
trump_questions = q1 + q2 + q3
trump_answers = a1 + a2 + a3

# Splitting training and validation sets

train_dataset = tf.data.Dataset.from_tensor_slices((questions[:40000], answers[:40000]))
val_dataset = tf.data.Dataset.from_tensor_slices((questions[40000:], answers[40000:]))

trump_dataset = tf.data.Dataset.from_tensor_slices((trump_questions, trump_answers))

def preprocess_text(sentence):
   sentence = tf.strings.lower(sentence)
   # Adding a space between the punctuation and the last word to allow better tokenization
   sentence = tf.strings.regex_replace(sentence, r"([?.!,])", r" \1 ")
   # Replacing multiple continous spaces with a single space
   sentence = tf.strings.regex_replace(sentence, r"\s\s+", " ")
   # Replacing non english words with spaces
   sentence = tf.strings.regex_replace(sentence, r"[^a-z?.!,]+", " ")
   sentence = tf.strings.strip(sentence)
   sentence = tf.strings.join(["[start]", sentence, "[end]"], separator=" ")
   return sentence

vectorizer = layers.TextVectorization(
   VOCAB_SIZE,
   standardize=preprocess_text,
   output_mode="int",
   output_sequence_length=MAX_LENGTH)

# We will adapt the vecotorizer to both the questions and answers
# This dataset is batched to parallelize and speed up the process
vectorizer.adapt(tf.data.Dataset.from_tensor_slices((questions + answers)).batch(128))

def vectorize_text(inputs, outputs):
   inputs, outputs = vectorizer(inputs), vectorizer(outputs)
   # One extra padding token to the right to match the output shape
   outputs = tf.pad(outputs, [[0, 1]])
   return (
      {"encoder_inputs": inputs, "decoder_inputs": outputs[:-1]},
      {"outputs": outputs[1:]})

train_dataset = train_dataset.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
trump_dataset = trump_dataset.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
trump_dataset = trump_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

class FNetEncoder(layers.Layer):
   def __init__(self, embed_dim, dense_dim, **kwargs):
      super(FNetEncoder, self).__init__(**kwargs)
      self.embed_dim = embed_dim
      self.dense_dim = dense_dim
      self.dense_proj = keras.Sequential([
         layers.Dense(dense_dim, activation="relu"),
         layers.Dense(embed_dim)])
      self.layernorm_1 = layers.LayerNormalization()
      self.layernorm_2 = layers.LayerNormalization()

   def get_config(self):
      return {"embed_dim": self.embed_dim, "dense_dim": self.dense_dim}

   def call(self, inputs):
      # Casting the inputs to complex64
      inp_complex = tf.cast(inputs, tf.complex64)
      # Projecting the inputs to the frequency domain using FFT2D and
      # extracting the real part of the output
      fft = tf.math.real(tf.signal.fft2d(inp_complex))
      proj_input = self.layernorm_1(inputs + fft)
      proj_output = self.dense_proj(proj_input)
      return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(layers.Layer):
   def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
      super(PositionalEmbedding, self).__init__(**kwargs)
      self.token_embeddings = layers.Embedding(
         input_dim=vocab_size, output_dim=embed_dim)
      self.position_embeddings = layers.Embedding(
         input_dim=sequence_length, output_dim=embed_dim)
      self.sequence_length = sequence_length
      self.vocab_size = vocab_size
      self.embed_dim = embed_dim

   def get_config(self):
      return {"sequence_length": self.sequence_length, "vocab_size": self.vocab_size,
         "embed_dim": self.embed_dim}

   def call(self, inputs):
      length = tf.shape(inputs)[-1]
      positions = tf.range(start=0, limit=length, delta=1)
      embedded_tokens = self.token_embeddings(inputs)
      embedded_positions = self.position_embeddings(positions)
      return embedded_tokens + embedded_positions
    
   def compute_mask(self, inputs, mask=None):
      return tf.math.not_equal(inputs, 0)

        
class FNetDecoder(layers.Layer):
   def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
      super(FNetDecoder, self).__init__(**kwargs)
      self.embed_dim = embed_dim
      self.latent_dim = latent_dim
      self.num_heads = num_heads
      self.attention_1 = layers.MultiHeadAttention(
         num_heads=num_heads, key_dim=embed_dim)
      self.attention_2 = layers.MultiHeadAttention(
         num_heads=num_heads, key_dim=embed_dim)
      self.dense_proj = keras.Sequential(
         [layers.Dense(latent_dim, activation="relu"),
          layers.Dense(embed_dim)])
      self.layernorm_1 = layers.LayerNormalization()
      self.layernorm_2 = layers.LayerNormalization()
      self.layernorm_3 = layers.LayerNormalization()
      self.supports_masking = True

   def get_config(self):
      return {"embed_dim": self.embed_dim, "latent_dim": self.latent_dim,
         "num_heads": self.num_heads}

   def call(self, inputs, encoder_outputs, mask=None):
      causal_mask = self.get_causal_attention_mask(inputs)
      if mask is not None:
         padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
         padding_mask = tf.minimum(padding_mask, causal_mask)
          
      attention_output_1 = self.attention_1(
         query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
      out_1 = self.layernorm_1(inputs + attention_output_1)

      attention_output_2 = self.attention_2(query=out_1,
                                            value=encoder_outputs,
                                            key=encoder_outputs,
                                            attention_mask=padding_mask)
      out_2 = self.layernorm_2(out_1 + attention_output_2)

      proj_output = self.dense_proj(out_2)
      return self.layernorm_3(out_2 + proj_output)
     
   def get_causal_attention_mask(self, inputs):
      input_shape = tf.shape(inputs)
      batch_size, sequence_length = input_shape[0], input_shape[1]
      i = tf.range(sequence_length)[:, tf.newaxis]
      j = tf.range(sequence_length)
      mask = tf.cast(i >= j, dtype="int32")
      mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
      mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1,1], dtype=tf.int32)],
                       axis=0)
      return tf.tile(mask, mult)
     
def create_model():
   encoder_inputs = keras.Input(shape=(None,), dtype="int32", name="encoder_inputs")
   x = PositionalEmbedding(MAX_LENGTH, VOCAB_SIZE, EMBED_DIM)(encoder_inputs)
   encoder_outputs = FNetEncoder(EMBED_DIM, LATENT_DIM)(x)
   encoder = keras.Model(encoder_inputs, encoder_outputs)
   decoder_inputs = keras.Input(shape=(None,), dtype="int32", name="decoder_inputs")
   encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name="decoder_state_inputs")
   x = PositionalEmbedding(MAX_LENGTH, VOCAB_SIZE, EMBED_DIM)(decoder_inputs)
   x = FNetDecoder(EMBED_DIM, LATENT_DIM, NUM_HEADS)(x, encoded_seq_inputs)
   x = layers.Dropout(0.5)(x)
   decoder_outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(x)
   decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs, name="outputs")
   decoder_outputs = decoder([decoder_inputs, encoder_outputs])
   fnet = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="fnet")
   return fnet


fnet = create_model()
fnet.compile(tf.keras.optimizers.Adam(), keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

try:
   fnet = tf.keras.models.load_model("fnet.h5", custom_objects={"PositionalEmbedding": PositionalEmbedding,
                                                                "FNetEncoder": FNetEncoder,
                                                                "FNetDecoder":FNetDecoder})

   vectorizer_data = pickle.load(open("vectorizer.save", "rb"))
   vectorizer.set_weights(vectorizer_data['weights'])
   vectorizer.from_config(vectorizer_data['config'])
except IOError:
   fnet.fit(train_dataset, epochs=50, validation_data=val_dataset)
   pickle.dump({'config': vectorizer.get_config(),
                'weights': vectorizer.get_weights()},
               open("vectorizer.save", "wb"))
   fnet.save('fnet.h5')

VOCAB = vectorizer.get_vocabulary()

trump = create_model()
trump.compile(tf.keras.optimizers.Adam(learning_rate=5e-5), keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
trump.set_weights(fnet.get_weights())

try:
   trump = tf.keras.models.load_model("trump.h5", custom_objects={"PositionalEmbedding": PositionalEmbedding,
                                                                  "FNetEncoder": FNetEncoder,
                                                                  "FNetDecoder":FNetDecoder})

   vectorizer_data = pickle.load(open("trump_vectorizer.save", "rb"))
   vectorizer.set_weights(vectorizer_data['weights'])
   vectorizer.from_config(vectorizer_data['config'])
except IOError:
   trump.fit(trump_dataset, epochs=100)
   pickle.dump({'config': vectorizer.get_config(),
                'weights': vectorizer.get_weights()},
               open("trump_vectorizer.save", "wb"))
   trump.save('trump.h5')
   
def decode_sentence(input_sentence):
   # Mapping the input sentence to tokens and adding start and end tokens
   tokenized_input_sentence = vectorizer(
      tf.constant("[start] " + preprocess_text(input_sentence) + " [end]"))
   # Initializing the initial sentence consisting of only the start token
   tokenized_target_sentence = tf.expand_dims(VOCAB.index("[start]"), 0)
   decoded_sentence = ""

   for i in range(MAX_LENGTH):
      # Get the predictions
      predictions = trump.predict(
         {
            "encoder_inputs": tf.expand_dims(tokenized_input_sentence, 0),
            "decoder_inputs": tf.expand_dims(tf.pad(
               tokenized_target_sentence,
               [[0, MAX_LENGTH - tf.shape(tokenized_target_sentence)[0]]]), 0)})
      # Calculating the token with maximum probability and getting the corresponding word
      sampled_token_index = tf.argmax(predictions[0, i, :])
      sampled_token = VOCAB[sampled_token_index.numpy()]
      # If sampled token is the end token then stop generating and return the sentence
      if tf.equal(sampled_token_index, VOCAB.index("[end]")):
         break
      decoded_sentence += sampled_token + " "
      tokenized_target_sentence = tf.concat(
         [tokenized_target_sentence, [sampled_token_index]], 0)

   return decoded_sentence


for i in ["Hello", "Can you introduce yourself?", "What's your name?", "Do you want to build a wall?", "What do you think about Joe Biden?"]:
   print(f"Linas: {i} \t\t Trump: {decode_sentence(i)}")
