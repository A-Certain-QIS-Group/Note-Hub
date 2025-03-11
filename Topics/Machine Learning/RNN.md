# RNN

## RNN
- hidden state: `h[t] = tanh(Linear(x[t], h[t-1]))`

## GRU
- hidden state: `h[t] = (1-z[t]) ⊙ h[t-1] + z[t] ⊙ ̃h[t]`
    - candidate: `̃h[t] = tanh(Linear(x[t], r[t] ⊙ h[t-1]))`
- update gate: `z[t] = σ(Linear(x[t],h[t-1]))`
- reset gate `r[t] = σ(Linear(x[t],h[t-1]))`


## LSTM
- output hidden: `h[t] = o[t] ⊙ tanh(c[t])`
- cell state: `c[t] = f[t] ⊙ c[t-1] + i[t] ⊙ ̃c[t]`
    - candidate: `̃c[t] = tanh(Linear(x[t],h[t-1]))`
- forgot gate: `f[t] = σ(Linear(x[t],h[t-1]))`
- input gate: `i[t] = σ(Linear(x[t],h[t-1]))`
- output gate: `o[t] = σ(Linear(x[t],h[t-1]))`

## Parallel Scan
- `x[t] = a[t]x[t-1] + b[t]` can be efficiently computed in parallel
- <https://arxiv.org/pdf/2311.06281>
- use log space for numerical stability
- $\log(x_t) = a^∗_t + \log(x_0+b^∗_t)$
- $a^∗_t = ∑\log a_t$
- $b^∗_t = ∑ \exp(\log b_t - a^∗_t)$


## minGRU
- interior state no dep on h, remove reset gate
- `h[t] = (1-z[t]) ⊙ h[t-1] + z[t] ⊙ ̃h[t]`
    - `̃h[t] = tanh(Linear(x[t]))`
- `z[t] = σ(Linear(x[t]))`

## minLSTM
- `h[t] = f'[t] ⊙ h[t-1] + i'[t] ⊙ ̃h[t]`
    - `̃h[t] = Linear(x[t])`
- `f[t], i[t] = σ(Linear(x[t]))`
- `f',i' = f/(f+i), i/(f+i)`

## parallel scan



