# Notes on Large Language Model

*Oct.26 2024* - *Oct.28 2024*

## References

[Youtube Animation by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
[Attention is All You Need Paper](https://arxiv.org/abs/1706.03762)

## Overview

- Language Model takes a sequence of words (tokens) and predict the conditional probability of the next word (token)
  - $P(\text{``The"},\text{``capital"},\text{``of"},\text{``the United States"},\text{``is"}|\text{``Washington, D.C."}) ≈ 1$
  - $P(\text{``The"},\text{``capital"},\text{``of"},\text{``the United States"},\text{``is"}|\text{``Chicago"}) ≈ 0$
- Trained on a large corpus collected from internet crawls
  - 10B-1T tokens
- By sampling the next word recursively, we can generate some texts
  - empirically, LLM shows ability of
    - understanding and reply to human language, with contexture awareness
    - summarizing factual knowledge from training data and memorizing it
    - *perhapes* basic reasoning skills
  - Out-of-Distribution (OOD) generalization
    - LLM is able to produce something not appeared in the training data
    - "If Einstein found himself on the moon, he will..."
- Deep Neural Network, trained by Backpropagation
- Outline of GPT and LLaMa Family Models Architecture
  - Word Embedding
  - Attention
  - Multiple-Layer Percepton
  - eignieering details

## Word Embedding

- Words (tokens) are labeled by integers $i$
  - in practice, we use subword instead of word as minimal unit. the division is done by the Tokenizer
  - vocabulary size ~50k
- The schematic meaning of words can be embedded into a vector space $ℋ$
  - $\vec E(\text{``king"}) - \vec E(\text{``male"}) + \vec E(\text{``female"}) ≈ \vec E(\text{``queen"})$
- Dot product detects certain schematic meaning
  - $\vec E(\text{``tall"})⋅\vec E(\text{``giraffe"})>0$
- Embedding matrix: maps words to a vector which have naive schematic meaning
  - $E=\begin{bmatrix}\vec E_0 & \vec E_1 & ...\end{bmatrix}$
  - the i-th column represents the word vector of the i-th word
- Embedding Dimension $d=\dim ℋ$: the dimension of schematic space
  - $d$ ~ 256(toy)-16384(GPT4-like)

#### Code

```python
input_tokens = tokenizer.encode("Hello, world!").ids
input_tokens = torch.tensor(input_tokens,)[None, :]
    # (bs, seqlen)
embedding = nn.Embedding(vocab_size, dim) 
word_vectors = embedding(input_tokens) 
    # (bs, seqlen, dim)
```

### There is more space than you think

- Johnson–Lindenstrauss lemma
  - In a $d$-dimensional vector space, one can write down $N=𝒪(e^{ϵ^2d})$ nearly linear independent "basis" which are $ϵ$-peperdendular to each other
- number of meanings LLM can comprehend scales exponentially with embedding dimension
  - we can host 50k vocabulary in a 256-dimensional embedding space
    - even excluding synonyms, the number of independent concepts are still much larger than 256
  - for larger models, it can host *exponentially* more nuanced schematic meanings
    - e.g. (fox) → (a (fox) that is (brown), (quick), and is ((jumpping) over a ((dog) that is (lazy))))
- Picture: the irrelevant noise for one is the relevant knowledge for some other

## Token Prediction

- The output of the Language Model is a word vector $\vec x$, representing the meaning of the next word
- $y_i = \vec E_i ⋅ \vec x$ are called logits
- the probability $p_i$ that the next word is $i$ is obtained through softmax
  - $\text{softmax}: y_i → p_i = e^{y_i/T} / ∑_j e^{y_j/T}$
  - temperature $T$ controls how random the output will be. when $T=0$ one just choose the

### Code

```python
class Transformer(nn.Module):
    @torch.inference_mode()
    def stream_generate(self, ids, max_new_tokens, temperature=1.0):
        for step in range(max_new_tokens):
            ids = ids[:, -self.params.max_seq_len:]
            logits = self(ids)
            last_logit = logits[:, -1, :] 
            probs = F.softmax(last_logit / temperature, dim=-1)
            id_next = torch.multinomial(probs, num_samples=1)
            ids = torch.cat((ids, id_next), dim=1)
            yield id_next
```

## Transformer Layers

- Motivation
  - The same token have different meaning in different contex
    - "American shrew **mole**" → Animal
    - "One **mole** of carbon dioxide" → Unit
    - "Take a biopsy of the **mole**" → Skin Issue
  - Each layer "transforms" the meaning of each token in the sentence by "decorating" it with other tokens in the sentence
    - Tower → Eiffel Tower → Miniature Eiffel Tower
      - on the second layer, "Tower" represents a very big architecture
      - on the third layer, "Tower" represents a tiny toy
- in each Transformer Layer
  - Attention sublayer propagages information between words
  - FeedForward sublayer synthesizes collected information
- number of layers
  - 6(toy)-more than 96(GPT4-like)

### Code

```python
class Transformer(nn.Module):
    def forward(self, tokens: torch.Tensor)->torch.Tensor:
        bs, seqlen = tokens.shape
        x = self.embedding(tokens)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x) # each layer is a TransformerBlock
        x = self.norm(x)
        logits = self.output(x) 
            # log probabilities of the next token
        return logits 
            # (batch_size, sequence_length, vocab_size)

class TransformerBlock(nn.Module):
    def forward(self, x):
        x = x + self.attention.forward(self.attention_norm(x))
        x = x + self.feed_forward.forward(self.ffn_norm(x))
        return x
```

## Attention

- Different words have different weights and ways when decorating other words
  - e.g., nouns look for adjectives, verbs look for adverbs
- Attention Mechanism
  - for each word in the sentence $\{\vec x_0, \vec x_1, \vec x_2...\}$
    - Key vectors $\vec K_t = W_K⋅\vec x_t ∈ ℋ_k$ describes how those words might help decorating other words
    - Query vectors $\vec Q_{t'} = W_Q⋅\vec x_{t'} ∈ ℋ_k$ represents what the word $t$ is looking for
    - The influence weight of word $t$ on word $t'≥t$ is estimated by the dot product $w_{tt'}=\vec Q_{t'}⋅\vec K_t$. the weight for each $t'$ is further normalized by softmax $w̃_{tt'} = e^{w_{tt'}}/∑_{s'} e^{w_{ts'}}$
  - Value vector $\vec V_{t} = W_V⋅\vec x_{t}  ∈ ℋ_v$ represents the part of meaning $t$ should contribute to $t'$
    - $\vec x'_{t'} = \vec x_t + W_O⋅∑_t w̃_{tt'} \vec V_{t'}$
    - $W_V$ transforms $\vec x_t$ to the value subspace $ℋ_v$ of dimension $d_v ≈ d_k$, then being projected back by $W_O$.
      - $W_O⋅W_V$ has lower rank and dof than a full $d×d$ matrix
- Multi-Head Attention
  - $n_h$ set of $W^{(α)}_Q$, $W^{(α)}_K$, $W^{(α)}_V$ presents (perhapes) different decoration mechanisms. each called an "attention head"
- Head Dimension $d_k=\dim ℋ_k$
  - to better matching information bandwidth during computation, empirically, we choose $d_v  n_h = d, d_k=d_v$
  - $d_k$ ~ 64(toy) - 128
  - $n_h$ ~ 8(toy) - 128(GPT4-like)
  - $d$ ~ 256(toy) - 16384(GPT4-like)

### Compat formular

- $X=\begin{bmatrix} \vec x_0 \\ \vec x_1 \\ ... \end{bmatrix}$ consists of input vectors
- $Q =\begin{bmatrix}\vec Q_0 \\ \vec Q_1 \\ ...  \end{bmatrix} = X⋅W_Q^T, K=X⋅W_K^T, V= X⋅W_V^T$
- $ \text{Attention}(Q,K,V) = \text{softmax}(\frac{1}{\sqrt{d_k}}(Q⋅K^T)+m) V$
  - $m_{t't}=\begin{cases} -∞, &  t' > t \\ 0, &\text{otherwise} \end{cases}$ masks out information propagation from future words to past words
  - $\text{softmax}: w_{tt'} ↦ w̃_{tt'} = e^{w_{tt'}}/∑_{s'} e^{w_{ts'}}$ normalizes the sum of weight to 1.
  - $\frac{1}{\sqrt{d_k}}$ is added for numerical stability
- $X' = X + ∑_α \text{Attention}(Q^{(α)},K^{(α)},V^{(α)})⋅W_O^{(α)T} $

### Code

```python
class Attention(nn.Module):
    def forward(self, x):
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk, xv = xq.view(bs, seqlen, n_heads, head_dim),...
        xq, xk = rotary_embedding(xq, xk) # explain later
        xq, xk, xv = xq.transpose(1,2),... 
            # (bs, n_heads, seqlen, head_dim)
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
            # upper-triangular mask with -inf values above the main diagonal
        scores = torch.matmul(xq, xk.transpose(2, 3)) 
        scores = scores/math.sqrt(head_dim) + mask[:, :, :seqlen, :seqlen]
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # (bs, n_heads, seqlen, seqlen)
        scores = F.dropout(scores, p=dropout_p) # explain later
        output = torch.matmul(scores, xv) 
            # (bs, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bs, seqlen, n_heads*head_dim)
        output = wo(output)
            # (bs, n_heads, seqlen, dim)
        output = self.resid_dropout(output) # explain later
        return output
```

## FeedForward Layer

- Inference additional meanings of each individual token using the information collected from the attention layer
  - "Michael" + "Jordan" → "Basketball"
  - "Michael" + "Jackson" → "Singer"
- Multiple Layer Perceptron
  - $FFN(\vec x) = W_2⋅σ( W_1⋅\vec x+ b_1 ) + b_2$
  - linear-transforms $\vec x$ to a higher-dimensional space $ℋ_\text{hidden}$, apply element-wise activation function $σ$ , then linear-transforms it back
- Rectified Linear Unit
  - $σ(x) = \text{max}(x,0)$
  - screens out negative x. x will only have influence if it is positive
- hidden dimension $d_\text{hidden} = \dim ℋ_\text{hidden}$
  - empirically, we choose $d_\text{hidden}=4 d$

### A handwaving explaination

- $W_1$ stores criterions of concepts to be detected
  - e.g. $W_1=\begin{bmatrix} \vec E(\text{Michael}) + \vec E(\text{Jordan}) \\ \vec E(\text{Michael}) + \vec E(\text{Jackson}) \end{bmatrix}, b_1=\begin{bmatrix} -1 \\ -1 \end{bmatrix}$
- Each row, follows by the ReLU gating, acts like a logical gate
  - e.g. $σ((\vec E(\text{Michael}) + \vec E(\text{Jordan}))⋅\vec x -1)$ returns 1 only when the token $\vec x$ was both decorated by "Michael" and "Jordan"
  - the output of each row is called a "neuron", the basic logical unit in neural network
- $W_2$ stores facts about the corresponding concepts in $ℋ_\text{hidden}$
  - e.g. $W_2=\begin{bmatrix} \vec E(\text{basketball}) + \vec E(\text{born in 1963}) + ... \\ \vec E(\text{singer}) + \vec E(\text{born in 1958}) + ... \end{bmatrix}$
  - updates inferred additional facts about the detected concepts on each token
  - "association memory"

### There is more knowledge than you think

- there are more than $d_\text{hidden}$ criteria-fact pairs
  - In practice, it is hardly the case that individual neuron represents independ concepts.
  - Johnson–Lindenstrauss lemma implies we can host $N ≫ d_\text{hidden}$ neuron activation patterns which are nearly linear independent to each other
  - $W_1$, $W_2$ stores the *superposition* of all those criterions
  - the knowledge we used in scenario B is hidden in the noise for tasks in scenario A.
- knowledge capacity scales with number of parameters
  - 2 bit / param [Physics of Lm 3.3](https://arxiv.org/abs/2404.05405)
    - empirically universal upper bound
    - achievable in ideal conditions

### Engineering Choices

- Sigmoid Linear Unit
  - $\text{Swish}(x) = x \times \text{sigmoid}(x) = \frac{x}{1+e^{-x}}$
    - provides gradient at negative x, avoid "dead neuron"
- Gated Linear Unit
  - $\text{GLU}(\vec x) = W_2 ⋅ (σ(W_1 ⋅ \vec x) ⊙ (W_3⋅ \vec x))$
    - $⊙$ is the element-wise multiplication

### Code

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

## Other Essential Components

### Rotary Embedding

- motivation
  - we want $\vec Q_t = W_Q ⋅ \vec x_t$ and $\vec K_t = W_K ⋅ \vec x_t$ contains information about the position $t$
  - we want the score $\vec  Q_{t'} ⋅ \vec K_t$ only depends on their relative position $t'-t$, not their absolute positions
- Rotary Embedding
  - $\begin{bmatrix}(\vec K_t)_{2i} \\ (\vec K_t)_{2i+1} \end{bmatrix} → \begin{bmatrix} cos(ω_i t) & -sin(ω_i t) \\ sin(ω_i t) & cos(ω_i t) \end{bmatrix} \begin{bmatrix}(\vec K_t)_{2i} \\ (\vec K_t)_{2i+1} \end{bmatrix}$
  - same for Q
  - rotate the every 2 dimension of xq and xk by their position in the sequence in different speed
  - different dimension rotates in different speed
    - $ω_i = Θ^{-\frac{2i}{d_h}}, 0≤2i<d_h$
  - exampe of the first few frequencies at $Θ=10000$, $d_h=64$
    - $1/ω_i =$ `[1.0, 1.3, 1.8, 2.4, 3.2, 4.2, 5.6, 7.5, 10.0, 13.3, 17.8,...]`

#### Code

```python
class RotaryEmbedding(nn.Module):
    # rotate the two halves of the vector. each pair of dimension is rotated at different speed
    def forward(self, q, k, t):
        bs, seq_len, n_heads, head_dim = q.shape
        freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2).float() / dim)).float()
            # need to user float32 in cos and sin
        freqs = self.freqs[None, None, None, :].float()
        freqs = torch.cat((freqs, freqs), dim=-1) 
        t = t[:, :, None, None].float()
        angles = freqs * t
        cos = angles.cos()
        sin = angles.sin()
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k
```

### RMSNorm

- rescales the vector by the activation of neurons
  - $\text{RMSNorm}(\vec x) = \frac{\vec x ⊙ \vec g }{\sqrt{\frac{1}{d}∑_i x_i x_i + ϵ}} $
  - only dependent on the current token, agnostic to other tokens or batch elements
  - do not shift the bias
  - gain each neuron individually through learnable parameter $\vec g$

#### Code

```python
class RMSNorm(torch.nn.Module):
    def forward(self, x):
        x=x.float() # need float32 precision
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * self.weight
        return x
```
