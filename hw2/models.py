import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
#import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical # for generate sequences

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).
    
    inputs: 
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



# Problem 1
class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.

  class RNN_Unit(nn.Module):
    def __init__(self, input_size, hidden_size):
      super(RNN.RNN_Unit, self).__init__()
      self.input_size = input_size
      self.hidden_size = hidden_size
      self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

    def init_weights(self, k = None):
        if k is None:
            k = 1. / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.i2h.weight, -1*k, k)
        nn.init.uniform_(self.i2h.bias, -1*k, k)

    def forward(self, input):
      return torch.tanh(self.i2h(input))


  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    """
    emb_size:     The numvwe of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at 
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the 
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    super(RNN, self).__init__()

    # TODO ========================
    # Initialization of the parameters of the recurrent and fc layers. 
    # Your implementation should support any number of stacked hidden layers 
    # (specified by num_layers), use an input embedding layer, and include fully
    # connected layers with dropout after each recurrent layer.
    # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding 
    # modules, but not recurrent modules.
    #
    # To create a variable number of parameter tensors and/or nn.Modules 
    # (for the stacked hidden layer), you may need to use nn.ModuleList or the 
    # provided clones function (as opposed to a regular python list), in order 
    # for Pytorch to recognize these parameters as belonging to this nn.Module 
    # and compute their gradients automatically. You're not obligated to use the
    # provided clones function.

    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.dp_keep_prob = dp_keep_prob
    self.i2e = nn.Embedding(vocab_size, emb_size) 
    self.h2o = nn.Linear(hidden_size, vocab_size)

    self.rnn_units = nn.ModuleList([])
    self.dropout = nn.Dropout(1 - dp_keep_prob)
    for i in range(num_layers):
      input_size = emb_size if i == 0 else hidden_size
      self.rnn_units.append(RNN.RNN_Unit(input_size, hidden_size))
      
    self.init_weights() # need this to initialize weights?
    return 

  def init_weights(self):
    # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
    # and the embedding and output biases to 0 (in place).
    # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly 
    # in the range [-k, k] where k is the square root of 1/hidden_size
    nn.init.uniform_(self.i2e.weight, -0.1, 0.1)
    nn.init.uniform_(self.h2o.weight, -0.1, 0.1)
    nn.init.zeros_(self.h2o.bias)
    k = 1. / math.sqrt(self.hidden_size)
    for i in range(self.num_layers):
      self.rnn_units[i].init_weights(k)
      

  def init_hidden(self):
    # initialize the hidden states to zero
    """
    This is used for the first mini-batch in an epoch, only.
    """
    hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
    return hidden.requires_grad_()
    # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

  def forward(self, inputs, hidden):
    # Compute the forward pass, using nested python for loops.
    # The outer for loop should iterate over timesteps, and the 
    # inner for loop should iterate over hidden layers of the stack. 
    # 
    # Within these for loops, use the parameter tensors and/or nn.modules you 
    # created in __init__ to compute the recurrent updates according to the 
    # equations provided in the .tex of the assignment.
    #
    # Note that those equations are for a single hidden-layer RNN, not a stacked
    # RNN. For a stacked RNN, the hidden states of the l-th layer are used as 
    # inputs to to the {l+1}-st layer (taking the place of the input sequence).

    """
    Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that 
                    represent the index of the current token(s) in the vocabulary.
                        shape: (seq_len, batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
    
    Returns:
        - Logits for the softmax over output tokens at every time-step.
              **Do NOT apply softmax to the outputs!**
              Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
              this computation implicitly.
                    shape: (seq_len, batch_size, vocab_size)
        - The final hidden states for every layer of the stacked RNN.
              These will be used as the initial hidden states for all the 
              mini-batches in an epoch, except for the first, where the return 
              value of self.init_hidden will be used.
              See the repackage_hiddens function in ptb-lm.py for more details, 
              if you are curious.
                    shape: (num_layers, batch_size, hidden_size)
    """
    seq_len = inputs.size(0) # (seq_len, batch_size)
    outputs_seqs = [] # list of output for each word in seq
    for i in range(seq_len):# iterate over seq
        embedded = self.dropout(self.i2e(inputs[i,:])) # (batch_size, emb_size)
        # size of each recurrent hidden layer output is (batch_size, hidden_size)
        output_cur_layer = None
        hs = [] # hidden states of each recurrent layer
        for j in range(self.num_layers): # more than 1 recurrent hidden layer
            input_cur_layer = embedded if j == 0 else output_cur_layer # 
            combined = torch.cat((input_cur_layer, hidden[j,:,:]), 1)
            hidden_cur_layer = self.rnn_units[j](combined) 
            output_cur_layer = self.dropout(hidden_cur_layer)
            hs.append(torch.unsqueeze(hidden_cur_layer, 0))
        
        output_cur_layer = self.h2o(output_cur_layer)
        hidden = torch.cat(hs, 0) # new hidden for next word in seq
        outputs_seqs.append(torch.unsqueeze(output_cur_layer, 0))
    
    logits = torch.cat(outputs_seqs, 0)
    return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden


  def generate(self, input, hidden, generated_seq_len):
    # TODO ========================
    # Compute the forward pass, as in the self.forward method (above).
    # You'll probably want to copy substantial portions of that code here.
    # 
    # We "seed" the generation by providing the first inputs.
    # Subsequent inputs are generated by sampling from the output distribution, 
    # as described in the tex (Problem 5.3)
    # Unlike for self.forward, you WILL need to apply the softmax activation 
    # function here in order to compute the parameters of the categorical 
    # distributions to be sampled from at each time-step.

    """
    Arguments:
        - input: A mini-batch of input tokens (NOT sequences!)
                        shape: (batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
        - generated_seq_len: The length of the sequence to generate.
                       Note that this can be different than the length used 
                       for training (self.seq_len)
    Returns:
        - Sampled sequences of tokens
                    shape: (generated_seq_len, batch_size)
    """
    samples = []
    predict = input # at time step 0, predict is input
    samples.append(torch.unsqueeze(predict, 0))
    for _ in range(generated_seq_len): # each seq index
        # output is the predicted word index, with batch_size elements
        embedded = self.dropout(self.i2e(predict)) # (batch_size, emb_size)
        output_cur_layer = None
        hs = []
        for j in range(self.num_layers): # more than 1 recurrent hidden layer
            input_cur_layer = embedded if j == 0 else output_cur_layer # 
            combined = torch.cat((input_cur_layer, hidden[j,:,:]), 1)
            hidden_cur_layer = self.rnn_units[j](combined) #(batch_size, hidden_size)
            output_cur_layer = self.dropout(hidden_cur_layer)
            hs.append(torch.unsqueeze(hidden_cur_layer, 0))
            
        output_cur_layer = self.h2o(output_cur_layer) 
        hidden = torch.cat(hs, 0) # new hidden for next word in seq
        p_output = torch.softmax(output_cur_layer / 100, dim = -1) # (batch_size, vocab_size)
        predict = Categorical(p_output).sample().long() # batch_size
        # another option is just give the predict with largest probability
        # output_cur_layer = torch.argmax(output_cur_layer, dim = -1) # word_index
        samples.append(torch.unsqueeze(predict, 0))
    
    samples = torch.cat(samples, 0)
    return samples


# Problem 2
class GRU(nn.Module): # Implement a stacked GRU RNN
  """
  Follow the same instructions as for RNN (above), but use the equations for 
  GRU, not Vanilla RNN.
  """
  class GRU_Unit(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU.GRU_Unit, self).__init__()
        self.input_size = input_size # = emd_size + hidden_size
        self.hidden_size = hidden_size
        self.i2r = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2z = nn.Linear(input_size + hidden_size, hidden_size)
        self.ir2hd = nn.Linear(input_size + hidden_size, hidden_size) # hd : h_tilde
      
    def init_weights(self, k = None):
        if k is None:
            k = 1. /math.sqrt(self.hidden_size)
        nn.init.uniform_(self.i2r.weight, -1*k, k)
        nn.init.uniform_(self.i2r.bias, -1*k, k)
        nn.init.uniform_(self.i2z.weight, -1*k, k)
        nn.init.uniform_(self.i2z.bias, -1*k, k)
        nn.init.uniform_(self.ir2hd.weight, -1*k, k)
        nn.init.uniform_(self.ir2hd.bias, -1*k, k)

    def forward(self, input):
        """input (batch_size, input_size+hidden_size)""" 
        x_size = self.input_size # real input size
        xt = input[:, :x_size] # (batch_size, x_size)
        ht = input[:, x_size:] # old ht (h_{t-1}) (batch_size, output_size)
        rt = torch.sigmoid(self.i2r(input)) # (batch_size, output_size)
        zt = torch.sigmoid(self.i2z(input)) # (batch_size, output_size)
        hd = torch.cat((xt, torch.mul(rt, ht)), 1) # (batch_size, input_size)
        hd = torch.tanh(self.ir2hd(hd)) # h_t^{tilde} (batch_size, input_size)
        ht_part1 = torch.mul(torch.sub(1, zt), ht)
        ht_part2 = torch.mul(zt, hd)
        ht = torch.add(ht_part1, ht_part2) # new ht (h_{t})
        return ht
      


  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
      super(GRU, self).__init__()
      self.emb_size = emb_size
      self.hidden_size = hidden_size
      self.seq_len = seq_len
      self.batch_size = batch_size
      self.vocab_size = vocab_size
      self.num_layers = num_layers
      self.dp_keep_prob = dp_keep_prob
      self.i2e = nn.Embedding(vocab_size, emb_size)  # input to embedding 
      self.h2o = nn.Linear(hidden_size, vocab_size)   # last hidden to output(y)
      self.dropouts = nn.ModuleList([nn.Dropout(1 - dp_keep_prob)])    # dropout
      self.gru_units = nn.ModuleList([])

      for i in range(num_layers):
          input_size = emb_size if i == 0 else hidden_size
          self.gru_units.append(GRU.GRU_Unit(input_size, hidden_size))
          self.dropouts.append(nn.Dropout(1-dp_keep_prob))

      self.init_weights_uniform() # need this to initialize weights?
      return 

  def init_weights_uniform(self):
      nn.init.uniform_(self.i2e.weight, -0.1, 0.1)
      nn.init.uniform_(self.h2o.weight, -0.1, 0.1)
      nn.init.zeros_(self.h2o.bias)
      k = 1. / math.sqrt(self.hidden_size)
      for i in range(self.num_layers):
          self.gru_units[i].init_weights(k)
      return

  def init_hidden(self):
      hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
      return hidden.requires_grad_()

  def forward(self, inputs, hidden):
      seq_len = inputs.size(0) # (seq_len, batch_size)
      outputs = []        # list of output for each word in seq
      for i in range(seq_len): # iterate over seq
          embedded = self.dropouts[0](self.i2e(inputs[i,:])) # (batch_size, emb_size)
          # size of each recurrent hidden layer output is (batch_size, hidden_size)
          output_cur_layer = None
          hs = [] # list of hidden states of each recurrent layer
          for j in range(self.num_layers): # more than 1 recurrent hidden layer
              input_cur_layer = embedded if j == 0 else output_cur_layer # 
              combined = torch.cat((input_cur_layer, hidden[j,:,:]), 1)
              hidden_cur_layer = self.gru_units[j](combined) 
              output_cur_layer = self.dropouts[j+1](hidden_cur_layer)
              hs.append(torch.unsqueeze(hidden_cur_layer, 0))
            
          output_cur_layer = self.h2o(output_cur_layer) # final output of model
          hidden = torch.cat(hs, 0) # new hidden for next word in seq
          # output for cur word in seq, no softmax
          outputs.append(torch.unsqueeze(output_cur_layer, 0))
        
      logits = torch.cat(outputs, 0)
      return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

  def generate(self, input, hidden, generated_seq_len):
      samples = []
      predict = input # at time step 0, predict is input
      samples.append(torch.unsqueeze(predict, 0))
      for i in range(generated_seq_len): # each seq index
          # output is the predicted word index, with batch_size elements
          embedded = self.dropouts[0](self.i2e(predict)) # (batch_size, emb_size)
          output_cur_layer = None
          hs = []
          for j in range(self.num_layers): # more than 1 recurrent hidden layer
              input_cur_layer = embedded if j == 0 else output_cur_layer # 
              combined = torch.cat((input_cur_layer, hidden[j,:,:]), 1)
              hidden_cur_layer = self.gru_units[j](combined) #(batch_size, hidden_size)
              output_cur_layer = self.dropouts[j+1](hidden_cur_layer)
              hs.append(torch.unsqueeze(hidden_cur_layer, 0))
            
          output_cur_layer = self.h2o(output_cur_layer)
          hidden = torch.cat(hs, 0) # new hidden for next word in seq
          p_output = F.softmax(output_cur_layer / 100, dim = -1) # (batch_size, vocab_size)
          predict = Categorical(p_output).sample().long() # batch_size
          # another option is just give the predict with largest probability
          # output_cur_layer = torch.argmax(output_cur_layer, dim = -1) # word_index
          samples.append(torch.unsqueeze(predict, 0))
    
      samples = torch.cat(samples, 0)
      return samples

# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



#----------------------------------------------------------------------------------

# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    # Based on: https://github.com/harvardnlp/annotated-transformer

    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units # hidden_size

        # TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout
        self.n_heads = n_heads
        self.linears = clones(nn.Linear(n_units, n_units), 4)
        #k = 1. / math.sqrt(self.n_units)
        #for linear in self.linears:
        #    nn.init.uniform_(linear.weight, -1*k, k)
        #    nn.init.uniform_(linear.bias, -1*k, k)
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax 
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.
        batch_size = query.size(0)
        Qi, Ki, Vi = [l(x).view(
            batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) 
            for l, x in zip(self.linears, (query, key, value))]
            # if the lengths in zip are different, zip returns short len of pair
        d_k = Qi.size(-1) # d_k = self.d_k
        scores = torch.matmul(Qi, Ki.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        Ai = torch.softmax(scores, dim = -1)
        Ai = self.dropout(Ai) # reasonable? dropout an probability?
        Hi = torch.matmul(Ai, Vi)
        Hi = Hi.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        A = self.linears[-1](Hi)
        return A # A shape: (batch_size, seq_len, self.n_units)

#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer
class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        # input_sequence shape: (batch_size, seq_len, vocab_size)
        # mask shape: (batch_size, seq_len, vocab_size, vocab_size)
        embeddings = self.embedding(input_sequence)
        # embedding shape: 
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. 
    input:  size, int
    output: mask, torch.tensor (1, size, size)
    Example: size = 4 will output:

    tensor([[[1, 0, 0, 0],
             [1, 1, 0, 0],
             [1, 1, 1, 0],
             [1, 1, 1, 1]]], dtype=torch.uint8)
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    # it seems that the above 2 lines are equivalent to next line:
    #return np.tril(np.ones(attn_shape), k = 0).astype('uint8')


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



import os
import collections


def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]
        
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word


def repackage_hidden(h):
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)        
        

def generate_seq(model, seq_len, batch_size, vocab_size):
    model.eval()
    print("generating text....")
    gen_hidden = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
    gen_hidden = repackage_hidden(gen_hidden)
    gen_input = torch.randint(0, vocab_size, (batch_size,)).contiguous().to(device)#.cuda()
    samples = model.generate(gen_input, gen_hidden, seq_len)
    # size of samples (seq_len, batch_size)
    samples = samples.transpose(0, 1).cpu().numpy()
    for i in range(batch_size): # for each batch
        print("sequence {}: ".format(i), end = " ")
        for j in range(seq_len):
            print(id_2_word[samples[i,j]], end = " ")
        print()


if __name__ == "__main__":
    # Use the GPU if you have one
    if torch.cuda.is_available():
        print("Using the GPU")
        device = torch.device("cuda") 
    else:
        print("WARNING: You are about to run on cpu, and this will likely run out \
        of memory. \n You can try setting batch_size=1 to reduce memory usage")
        device = torch.device("cpu")
    #################################################
    ## build a model 
    data = "./data/"
    print('Loading data from ' + data)
    raw_data = ptb_raw_data(data_path=data)
    train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
    vocab_size = len(word_to_id)
    print('  vocabulary size: {}'.format(vocab_size))
    batch_size = 20
    seq_len = 30
    hidden_size = 200
    num_layers = 2
    emb_size = 100
    dp_keep_prob = 0.35

    model = RNN(batch_size = batch_size, 
                seq_len = seq_len, 
                hidden_size = hidden_size, 
                num_layers = num_layers,
                dp_keep_prob = 0.35, 
                emb_size = emb_size, 
                vocab_size = vocab_size)

    model.to(device)
    # invoke method of "generate" to produce a sequence by a model with initial
    # parameters
    gen_seq_len = 20
    gen_batch_size = 2
    generate_seq(model, gen_seq_len, gen_batch_size, vocab_size) 

   