# Training.

https://vxnuaj.com/infinity/word2vec

<br/>
I'll be training word2vec on the PTB dataset. The dataset comes with a mixture of train, validation, and test splits, but for the purposes of generating embeddings, I'll be merging all into one large training set.
<br/>

<br/>

```python {className="small-code"}
lines = 0

with open('data/train.txt', 'r') as f:

    for line in f:

        lines += 1

print(lines)
```
<br/>

```
> 49199
```
<br/>
We have about 49199 unique sequences in the document.

```python

from collections import Counter

with open('data/train.txt', 'r') as f:
    lines = f.readlines()

words = [word for line in lines for word in line.split()]
word_counts = Counter(words)

threshold = 10
processed_lines = []
for line in lines:
    processed_words = [
        word if word_counts[word] >= threshold else "_unk_"
        for word in line.split()
    ]

    processed_lines.append(" ".join(processed_words))

with open('data/trainv2.txt', 'w') as f:
    f.writelines([line + '\n' for line in processed_lines])
```
<br/>

Now my data is written in `trainv2.txt`.

<br/>
I'm simply going to be training the model using Gensim's API.

<br/>
Albeit, not the greatest since we can't see intermediate loss values as effectively (seemingly broken), it'll still work for lightweight training.

<br/>

I attempt to call loss values using ```LossCallback(CallBackAny2Vec)``` object, but consistently returned loss values of $0$.

<br/>

```python
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

class LossCallback(CallbackAny2Vec):
    def __init__(self, model):
        self.losses = []
        self.model = model
        self.epochs = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        epoch = len(self.losses)
        self.epochs.append(epoch)
        
        print(f"Epoch {epoch}: Loss = {loss}")
        

    def save_model(self):
        self.model.save("word2vec_model.model")
        print("Model saved to 'word2vec_model.model'")

def load_data(path):
    sequences = []
    with open(path, 'r') as f:
        for sequence in f:
            sequence = sequence.split()
            sequences.append(sequence)
    return sequences

sequences = load_data('data/trainv2.txt')

# Hyperparameters
embed_dim = 100
window_size = 5
epochs = 500

print("Initializing Model")
model = Word2Vec(
    sentences=sequences,
    vector_size=embed_dim,  
    window=window_size,     
    min_count=2,            
    sg=0,                   
    epochs=epochs,
    compute_loss = True
)

print('building vocab')
model.build_vocab(sequences)
print(f"Model initialized with {len(model.wv.index_to_key)} unique words")

loss_callback = LossCallback(model)

print("Training")

model.train(sequences, 
    total_examples=model.corpus_count, 
    epochs=epochs, 
    callbacks=[loss_callback]
    )

loss_callback.save_model()
```
<br/>

We can now assess the cosine similarity for our $300$ dimensional embeddings. 
```math

\cos\theta = \frac{\vec{a}\vec{b}}{||\vec{a}|| \cdot ||\vec{b}||}

```
<br/>
Note that $\cos \theta \in [-1,1]$, where $1$ indicates high similarity, $0$ indicates orthogonal vectors, and $-1$ indicates dissimilar vectors.
<br/>
Here are the $10$ nearest vectors to the vector corresponding to $\text{computer}$.


<br/>

```python
print(model.wv.most_similar('computer', topn=10))
```
<br/>

```python
[('computers', 0.6619974374771118),
 ('software', 0.5994142889976501),
 ('digital', 0.47143417596817017),
 ('minicomputers', 0.44772008061408997),
 ('machines', 0.4456964433193207),
 ('supercomputer', 0.4447154998779297),
 ('toy', 0.444561630487442),
 ('ibm', 0.43565645813941956),
 ('equipment', 0.4339454472064972),
 ('electronic', 0.4326600134372711)]

```
<br/>

and purely for `computer` and `supercomputer`, done manually for comparison with Gensim's API (rounding errors??)

<br/>

```python
computer_vector = word_vectors.get_vector('computer')
supercomputer_vector = word_vectors.get_vector('supercomputer')

cos_sim = np.dot(computer_vector, supercomputer_vector) / (np.linalg.norm(computer_vector) * np.linalg.norm(supercomputer_vector))
print(cos_sim)
```

<br/>

```python
>> 0.44471553

```
<br/>

This puts our two vectors at an angle of about $60°$. Albeit this might indicate that semantic meaning hasn't been effectively captured in the embedding space, note that we're operating $\in \mathbb{R}^{300}$. The curse of dimensionality is very real.
<br/>
Let's project our embeddings onto a lower dimensionality, say $\mathbb{R}^3$.
<br/>
Rather than relying on linear methods, like $\text{PCA}$, I'll be using $\text{T-SNE}$ as it preserves the underlying non-linear structure of the manifold corresponding to the data.
<br/>
Just to keep things simple, I'll be keeping the perplexity value at $30$.
<br/>

```python
from sklearn.manifold import TSNE

word_vectors_reduced = TSNE(n_components=3,
                            verbose = 2).fit_transform(np_word_vectors)
```
<br/>

```bash
> [t-SNE] Computing 91 nearest neighbors...
> [t-SNE] Indexed 7393 samples in 0.001s...
> [t-SNE] Computed neighbors for 7393 samples in 0.293s...
> [t-SNE] Computed conditional probabilities for sample 1000 / 7393
> [t-SNE] Computed conditional probabilities for sample 2000 / 7393
> [t-SNE] Computed conditional probabilities for sample 3000 / 7393
> [t-SNE] Computed conditional probabilities for sample 4000 / 7393
> [t-SNE] Computed conditional probabilities for sample 5000 / 7393
> [t-SNE] Computed conditional probabilities for sample 6000 / 7393
> [t-SNE] Computed conditional probabilities for sample 7000 / 7393
> [t-SNE] Computed conditional probabilities for sample 7393 / 7393
> [t-SNE] Mean sigma: 7.943408
> [t-SNE] Computed conditional probabilities in 0.102s
> [t-SNE] Iteration 50: error = 138.7195740, gradient norm = 0.1113852 (50 iterations in 4.987s)
> [t-SNE] Iteration 100: error = 169.5654755, gradient norm = 0.0215915 (50 iterations in 3.936s)
> [t-SNE] Iteration 150: error = 169.5790863, gradient norm = 0.0794567 (50 iterations in 5.081s)
> [t-SNE] Iteration 200: error = 172.6365662, gradient norm = 0.0205027 (50 iterations in 5.441s)
> [t-SNE] Iteration 250: error = 175.0137177, gradient norm = 0.0084834 (50 iterations in 5.060s)
> [t-SNE] KL divergence after 250 iterations with early exaggeration: 175.013718
> [t-SNE] Iteration 300: error = 10.0660753, gradient norm = 0.0125573 (50 iterations in 4.626s)
> [t-SNE] Iteration 350: error = 9.2830153, gradient norm = 0.0105829 (50 iterations in 4.448s)
> [t-SNE] Iteration 400: error = 8.7625418, gradient norm = 0.0095544 (50 iterations in 4.570s)
> [t-SNE] Iteration 450: error = 8.5203323, gradient norm = 0.0089769 (50 iterations in 4.765s)
> [t-SNE] Iteration 500: error = 8.3215771, gradient norm = 0.0083913 (50 iterations in 5.046s)
> [t-SNE] Iteration 550: error = 8.1770744, gradient norm = 0.0079914 (50 iterations in 4.962s)
> [t-SNE] Iteration 600: error = 8.0532246, gradient norm = 0.0076157 (50 iterations in 4.933s)
> [t-SNE] Iteration 650: error = 7.9405279, gradient norm = 0.0073834 (50 iterations in 5.477s)
> [t-SNE] Iteration 700: error = 7.8365841, gradient norm = 0.0070329 (50 iterations in 6.343s)
> [t-SNE] Iteration 750: error = 7.7487297, gradient norm = 0.0067275 (50 iterations in 4.957s)
> [t-SNE] Iteration 800: error = 7.6637888, gradient norm = 0.0065605 (50 iterations in 5.399s)
> [t-SNE] Iteration 850: error = 7.5861931, gradient norm = 0.0063698 (50 iterations in 6.829s)
> [t-SNE] Iteration 900: error = 7.5195761, gradient norm = 0.0060594 (50 iterations in 8.860s)
> [t-SNE] Iteration 950: error = 7.4658055, gradient norm = 0.0058960 (50 iterations in 9.928s)
> [t-SNE] Iteration 1000: error = 7.4145055, gradient norm = 0.0057492 (50 iterations in 9.600s)
> [t-SNE] KL divergence after 1000 iterations: 7.414505
```
<br/>

Now computing cosine similarity for $\text{supercomputer}$ and $\text{computer}$
<br/>
```python
supercomputer_idx = word_vectors.get_index('supercomputer')
computer_idx = word_vectors.get_index('computer')
print(f"Cosine Similarity between supercomputer and computer: \
    {np.dot(word_vectors_reduced[supercomputer_idx], word_vectors_reduced[computer_idx]) \
    / (np.linalg.norm(word_vectors_reduced[supercomputer_idx]) \
    * np.linalg.norm(word_vectors_reduced[computer_idx]))}")
```
<br/>

```python
> Cosine Similarity between supercomputer and computer: 0.9849855899810791

```
<br/>
It's clear to see that the $\cos$ similarity works more effective over smaller $D$-dimensional spaces

<br/>

Intuitively, you can see this as a lower $D$ dimensional space containing more constraint on vectors in terms of directions they can point in, such that they're forced to be more tightly packed together.

<br/>

Of course, with similar vectors in the original $\mathbb{R}^D$, similarity becomes easier to see.
<br/>

Though, after reducing dimensionality, preserving the underlying structure can be a challenge the lower $D$ becomes.
<br/>

You can see this effect on my embeddings [here](https://projector.tensorflow.org/?config=https://www.vxnuaj.com/labspace/embeds.json), where it appears that (semantically) non-similar vectors appear to have a high cosine similarity despite being very dissimilar in the original $\mathbb{R}^{300}$ space.
<br/>

> If you load the link, make sure to toggle "Spherize Data" and disable PCA / Keep it to 3 components to properly visualize.