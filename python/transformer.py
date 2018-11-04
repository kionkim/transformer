#!/usr/bin/python

from IPython.display import Image
import copy, logging, math, time
import mxnet as mx
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn, utils, loss
from mxnet.ndarray import linalg_gemm2 as gemm2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('../log/transformer.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

#logger.info('logger test - info')
#logger.debug('logger test - debug')

ctx = mx.gpu()

def dummy():
    logger.info('test info')
    logger.debug('test debug')


def data_gen(V, batch, nbatches, seq_len, ctx = mx.cpu()):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = nd.array(np.random.randint(1, V, size=(batch, seq_len)))
        data[:, 0] = 1
        src, tgt = data, data
        yield Batch(src, tgt, 0)

    
# Just calculate attention context and attention
def attention(query, key, value, mask = None, dropout = None):
    """
    Compute scaled dot product attention
    """
    #logger.info('logger info in attention')
    #logger.debug('logger debug in attention')
    d_k = query.shape[-1]
    scores = gemm2(query, key, transpose_b = True) / math.sqrt(d_k) 
    p_attn = nd.softmax(scores, axis = -1)
    if dropout is not None:
        p_atten = dropout(p_attn)
    return gemm2(p_attn, value), p_attn


class MultiHeadedAttention(nn.Block):
    def __init__(self, h, d_model, dropout = .1):
        """
        Take in model size and number of heads.
        h: number of heads
        d_model: size of latent space for multihead self attention
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        with self.name_scope():
            self.attn = []
            self.linear_q = nn.Dense(d_model, in_units = d_model, flatten = False)
            self.linear_k = nn.Dense(d_model, in_units = d_model, flatten = False)
            self.linear_v = nn.Dense(d_model, in_units = d_model, flatten = False)
            self.linear_o = nn.Dense(d_model, in_units = d_model, flatten = False)
        
    def forward(self, query, key, value, mask = None):
        '''
        query: nd.array of size (batch, in_seq_len, embedding_dim)
        key: nd.array of size (batch, in_seq_len, embedding_dim)
        value: nd.array of size (batch, in_seq_len, embedding_dim)
        '''
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.expand_dims(axis = 1)
        batch = query.shape[0]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.linear_q(query)  
        key = self.linear_k(key)
        value = self.linear_v(value)
        # Transform to (Batch size * H * d_model * Filters) 
        # I.e., H headers of dimension d_model * Filters
        # Source and target sequences may differ in length. 
        # If we fix in_seq_len, there will be an error in Attention layer
        query = query.reshape((batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
        key = key.reshape((batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
        value = value.reshape((batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
        
        # 2) Apply attention on all the projected vectors in batch.
        x = []
        for i in range(self.h):
            _x, _attn = attention(query[:, i, :, :]
                                     , key[:, i, :, :]
                                     , value[:, i, :, :]
                                     , mask = mask
                                     , dropout = self.dropout)
            x.append(_x)
            self.attn.append(_attn)
        # 3) "Concat"  
        x = nd.concat(*x, dim = 2)
        # 4) Linear transofrmation
        x = self.linear_o(x)
        return x

    
class MultiHeadedAttention(nn.Block):
    def __init__(self, h, d_model, dropout = 0):
        """
        Take in model size and number of heads.
        h: number of heads
        d_model: size of latent space for multihead self attention
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.dense_layers = []
        with self.name_scope():
            self.attn = []
            for i in range(4):
                dense = nn.Dense(d_model, in_units = d_model, flatten = False)
                self.register_child(dense)
                self.dense_layers.append(dense)

    def forward(self, query, key, value, mask = None):
        '''
        query, key, value: nd.array of size (batch, in_seq_len, embedding_dim)
        '''
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.expand_dims(axis = 1)
        batch = query.shape[0]

        res = [self.dense_layers[i](x) for i, x in enumerate([query, key, value])]
        query, key, value = [x.reshape((batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3)) for x in res]

        x = []
        for i in range(self.h):
            _x, _attn = attention(query[:, i, :, :]
                                     , key[:, i, :, :]
                                     , value[:, i, :, :]
                                     , mask = mask
                                     , dropout = self.dropout)
            x.append(_x)
            self.attn.append(_attn)
        x = nd.concat(*x, dim = 2)
        x = self.dense_layers[3](x)
        x = x.reshape((batch, -1, self.h * self.d_k))
        return x

    
class Generator(nn.Block):
    """
    Define standard linear + softmax generation step.
    Apply generator per time step
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.vocab = vocab
        with self.name_scope():
            self.proj = nn.Dense(vocab, in_units = d_model)
        
    def forward(self, x):
        return nd.softmax(self.proj(x))

    
class Embeddings(nn.Block):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        with self.name_scope():
            self.emb = nn.Embedding(vocab, d_model)
            
        
    def forward(self, x):
        # nd.array of size (1, 1) => np.sqrt not nd.sqrt
        return self.emb(x) * np.sqrt(self.d_model)
    
    
class LayerNorm(nn.HybridBlock):
    """
    Constuct a layernorm module
    features: tuple (batch * in_seq_len * embedding_dim)
    """
    def __init__(self, d_model, eps = 1e-6): # Number of layers
        super(LayerNorm, self).__init__()
        self.eps = eps
        with self.name_scope():
            self.a = self.params.get('a', shape = (1, d_model), allow_deferred_init = False)
            self.b = self.params.get('b', shape = (1, d_model), allow_deferred_init = False)
        
    def hybrid_forward(self, F, x, a, b):
        mean = x.mean(axis = -1) # batch * _in_seq_len
        _mean = nd.repeat(mean.expand_dims(axis = -1), repeats = x.shape[-1], axis = -1) # batch * _in_seq_len * embedding_dim
        std = nd.sqrt(nd.sum(nd.power((x - _mean), 2), axis = -1) / x.shape[1]) # batch * _in_seq_len
        _std = nd.repeat(std.expand_dims(axis = -1), repeats = x.shape[-1], axis = -1) # batch * _in_seq_len * embedding_dim
        return F.elemwise_div(F.multiply((x - _mean), a), (_std  + self.eps)) + b

class LayerNorm(nn.HybridBlock):
    def __init__(self, d_model, eps = 1e-6): # Number of layers
        super(LayerNorm, self).__init__()
        self.eps = eps
        with self.name_scope():
            self.a = self.params.get_constant('a', nd.ones((1, d_model)))
            self.b = self.params.get_constant('b', nd.zeros((1, d_model)))
        
    def hybrid_forward(self, F, x, a, b):
        #print('x size = {}'.format(x))
        mean = x.mean(axis = -1) # batch * _in_seq_len
        _mean = F.repeat(mean.expand_dims(axis = -1), axis = -1, repeats = x.shape[-1])
        std = F.sqrt(F.sum(F.power((x - _mean), 2), axis = -1) / (x.shape[-1] - 1)) # batch * _in_seq_len
        _std = F.repeat(std.expand_dims(axis = -1), axis = -1, repeats = x.shape[-1]) # batch * _in_seq_len * embedding_dim
        return F.elemwise_div((x - _mean) * a, (_std  + self.eps)) + b    
    
class SublayerConnection(nn.Block):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    size: tuple (batch * in_seq_len * embedding_dim)
    """
    
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        with self.name_scope():
            self.norm = LayerNorm(size)
            

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        x: nd.array of size (batch * in_seq_len * embedding_dim)
        """
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Block):
    """
    Implements FFN equation
    d_model: size of latent space for multihead self attention
    hidden_dim: Hidden space for feedforward network
    """
    
    def __init__(self, d_model, hidden_dim, dropout = .1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Dense(hidden_dim, in_units= d_model, flatten = False)
        self.w_2 = nn.Dense(d_model, in_units = hidden_dim, flatten = False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Point-wise feed-forward network does not have in_seq dimension
        # We have to apply the same network for each time sequence
        return self.w_2(self.dropout(nd.relu(self.w_1(x))))

    
class PositionalEncoding(nn.Block):
    """
    Implement the PE function
    단순히 위치만 표현하고 싶음. 원래의 데이터 + 그 데이터가 해당하는 위치까지의 각 차원에서의 sine값과 cosine값
    d_model: size of latent space for multihead self attention
    """
    def __init__(self, d_model, dropout, max_len = 5000, ctx = mx.cpu()):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Compute the positional encodings once in log space.
        self.pe = nd.zeros((max_len, d_model), ctx = ctx)
        position = nd.arange(0, max_len).expand_dims(axis = 1)
        div_term = nd.exp(nd.arange(0, d_model, 2) * -(nd.log(nd.array([10000.0])) / d_model))
        self.pe[:, 0::2] = nd.sin(position * div_term) # 0부터 2번째마다. 그러니깐 0부터 1칸씩 떼서 한줄씩
        self.pe[:, 1::2] = nd.cos(position * div_term) # 1부터 2번째마다. 그러니깐 1부터 1칸씩 떼서 한줄씩
        self.pe = self.pe.expand_dims(axis = 0)
        
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)

    
class EncoderLayer(nn.Block):
    """
    Encoder is made up of self-attention nad feed forward
    size: size for sublayer connectioh tuple: (batch * in_seq_len * embedding_dim)
    self_attn: self attention layer
    feed_forward: network
    dropout: dropout rate float
    """
    def __init__(self, size, self_attn, feed_forward, dropout = .1):
        super(EncoderLayer, self).__init__()
        with self.name_scope():
            self.self_attn = self_attn
            self.feed_forward = feed_forward
            # Size is necessary for layer_normalization
            self.sublayer_0 = SublayerConnection(size, dropout)
            self.sublayer_1 = SublayerConnection(size, dropout)
            self.size = size
        
    def forward(self, x, mask = None):
        """
        Follow figure 1 for connections
        x: ndarray tuple (n_batch, in_seq_len, embedding_dim)
        """
        x = self.sublayer_0(x, lambda x: self.self_attn(x, x, x, mask))
        # Apply the same position-wise feed-forward network
        # Reshape (Batch size * in_seq_len * d_model) into (-1, d_model)
        return self.sublayer_1(x, self.feed_forward)


# We have to use sequential instead of 'list of layers' for parameter initialization
class Encoder(nn.Block):
    """
    Core encoder is a stack of N layers
    """
    def __init__(self, in_seq_len, d_model, h, d_hidden_feed_forward , N, dropout):
        super(Encoder, self).__init__()
        with self.name_scope():
            self.layers = nn.Sequential()
            self.norm = LayerNorm(d_model)
            for i in range(N): # stack of N layers
                self.layers.add(EncoderLayer(d_model
                          , MultiHeadedAttention(h, d_model)
                          , PositionwiseFeedForward(d_model, d_hidden_feed_forward)
                          , dropout
                          )
            )
            
    def forward(self, x, mask = None):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    
class DecoderLayer(nn.Block):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    """
    
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer_0 = SublayerConnection(size, dropout)
        self.sublayer_1 = SublayerConnection(size, dropout)
        self.sublayer_2 = SublayerConnection(size, dropout)
        
    def forward(self, x, memory, src_mask = None, trg_mask = None):
        """
        Follow Figure 1 (right) for connections.
        x: ndarray tuple (n_batch, in_seq_len, embedding_dim)
        memory: attention
        """
        m = memory
        x = self.sublayer_0(x, lambda x: self.self_attn(x, x, x, trg_mask))
        x = self.sublayer_1(x, lambda x: self.src_attn(x, m, m, src_mask)) # Concat information from input & target
        return self.sublayer_2(x, self.feed_forward)
        

class Decoder(nn.Block):
    """
    Generic N layer decoder with masking.
    """
    def __init__(self, in_seq_len, d_model, h, d_hidden_feed_forward, N, dropout):
        super(Decoder, self).__init__()
        with self.name_scope():
            self.norm = LayerNorm(d_model)
            self.layers = nn.Sequential()
            for i in range(N):
                self.layers.add(DecoderLayer(d_model
                              , MultiHeadedAttention(h, d_model)
                              , MultiHeadedAttention(h, d_model)
                              , PositionwiseFeedForward(d_model, d_hidden_feed_forward)
                              , dropout))
            
    def forward(self, x, memory, src_mask = None, trg_mask = None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, trg_mask)
        return self.norm(x)


def subsequent_mask(size):
    """
    Maks out subsequent positions.
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k = 1)#.astype('uint8')
    return nd.array(subsequent_mask) == 0

class EncoderDecoder(nn.Block):
    """
    A standard Encoder-Decoder architecture.
    """
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator
        
    def forward(self, src, trg, src_mask = None, trg_mask = None):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, trg, trg_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, trg, trg_mask):
        x = self.decoder(self.trg_embed(trg), memory, src_mask, trg_mask)
        # Transform into two dimensional array
        _x = x.reshape(-1, x.shape[-1])
        _x = self.generator(_x)
        _x = _x.reshape(x.shape[0], x.shape[1], self.generator.vocab)
        return _x

def make_model(src_vocab, trg_vocab, in_seq_len, out_seq_len, N = 6, d_model = 512, d_ff = 2048, h = 8, dropout = .1, ctx = mx.cpu()):
    """
    Helper: Construct a model from hyperparameters.
    """
    enc = Encoder(in_seq_len, d_model, h, d_ff, N, dropout)
    dec = Decoder(out_seq_len, d_model, h, d_ff, N, dropout)

    gen = Generator(d_model, trg_vocab)
    
    src_emb = nn.Sequential()
    src_emb.add(Embeddings(d_model, src_vocab))
    src_emb.add(PositionalEncoding(d_model, dropout, ctx = ctx))
    
    trg_emb = nn.Sequential()
    trg_emb.add(Embeddings(d_model, trg_vocab))
    trg_emb.add(PositionalEncoding(d_model, dropout, ctx = ctx))
    
    model = EncoderDecoder(enc, dec, src_emb, trg_emb, gen)
    return model


class Batch:
    """
    Object for holding a batch of data with mask during training
    src: 
    """
    
    def __init__(self, src, trg = None, pad = 0, ctx = mx.cpu()):
        self.src = src
        self.src_mask = (src != pad).expand_dims(axis = -2)
        self.ctx = ctx
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad, self.ctx)
            self.ntokens = nd.sum(self.trg_y != pad)
            
    @staticmethod
    def make_std_mask(trg, pad, ctx):
        """
        Create a mask to hide padding ad future words.
        Compare each element of trg_mask and sub_mask. (1, 1) -> 1 o.w. -> 0
        There is no bitwise operator for mxnet
        """
        trg_mask = (trg != pad).expand_dims(axis = -2)
        trg_mask = nd.repeat(trg_mask, repeats = trg_mask.shape[-1], axis = -2)
        sub_mask = subsequent_mask(trg.shape[-1])
        sub_mask = nd.repeat(sub_mask, repeats = trg_mask.shape[0], axis = 0)
        trg_mask = nd.multiply(trg_mask, sub_mask.as_in_context(ctx))
        return trg_mask
    
    
def run_epoch(data_iter, model, trainer, loss_fn, ctx = mx.cpu()):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    tokens = 0
    total_loss = 0

    for i, batch in enumerate(data_iter):        
        src = batch.src.as_in_context(ctx)
        trg = batch.trg.as_in_context(ctx)
        src_mask = batch.src_mask.as_in_context(ctx)
        trg_mask = batch.trg_mask.as_in_context(ctx)
        trg_y = batch.trg_y.as_in_context(ctx)
        ntokens = batch.ntokens
        with autograd.record():
            out = model(src, trg, src_mask, trg_mask)
            _out = out.reshape(-1, out.shape[-1])
            _cols = list(batch.trg_y.reshape(-1).asnumpy())
            _rows = list(range(_out.shape[0]))
            _idx = nd.array([_rows, _cols], ctx = ctx)
            _trg = nd.scatter_nd(nd.ones_like(trg_y.reshape(-1)), _idx, _out.shape)
            loss = nd.sum(loss_fn(_out, _trg))
            #print('_trg_onehot = {}'.format(_trg))
            #print('trg = {}'.format(trg_y[0]))
            loss.backward()
        trainer.step(out.shape[0])
        total_loss += loss.asnumpy()[0]
        total_tokens += ntokens.asnumpy()[0]
        tokens += ntokens.asnumpy()[0]
        if i % 50 == 0:
            elapsed = time.time() - start
            logger.info("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss.asnumpy()[0] / ntokens.asnumpy()[0], tokens / elapsed))
            print('--------')
            print('loss = {}'.format(loss.asnumpy()))
            print('_pred = {}'.format(           nd.argmax(_out, axis = 1)[:9].asnumpy()))
            print('_trg_recover = {}'.format(nd.argmax(_trg, axis = 1)[:9].asnumpy()))
            print('--------')
            start = time.time()
            tokens = 0
    return total_loss #/ total_tokens


if __name__ == "__main__":
    # Task: copy 10 input integers
    V = 20
    batch = 30
    n_batch = 20
    in_seq_len = 10
    out_seq_len = 9
    dropout = .1
    n_epoch = 10
    data = data_gen(V, batch, n_batch, in_seq_len)
    model = make_model(V, V, in_seq_len, out_seq_len, N=2, dropout = .1, ctx = ctx)
    model.collect_params().initialize(mx.init.Xavier(), ctx = ctx)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 1e-2, 'beta1': 0.9, 'beta2': 0.98 , 'epsilon': 1e-9})
    loss = gluon.loss.KLDivLoss(from_logits = False)
    
    
    for epoch in range(n_epoch):
        run_epoch(data_gen(V, batch, n_batch, in_seq_len, ctx = ctx), model, trainer, loss, ctx = ctx)
