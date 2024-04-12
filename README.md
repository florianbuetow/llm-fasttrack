# LLM-FastTrack

This is where I'm keeping track of everything I learn about Large Language Models.
It's straightforward – notes, code, and links to useful resources.

## What's Inside

- **Notes:** Quick thoughts, summaries, and explanations I've written down to better understand LLM concepts.
- **Code:** The actual code I've written while experimenting with LLMs. It's not always pretty, but it works (mostly).
- **Resources:** Links to articles, papers, and tutorials that have cleared things up for me. No fluff, just the good stuff.

## Why This Repo

I needed somewhere to dump my brain as I dive into LLMs. Maybe it'll help someone else, maybe not. But it's helping me keep track of my progress and organize my thoughts.

Feel free to look around if you're into LLMs or just curious about what I'm learning. No promises, but you might find something useful.

---

# Studyplan

This is the curriculum I'm following to learn about Large Language Models. It's a mix of PyTorch basics, LLM concepts, and real-world applications.
The first draft of the study plan has been generated by a LLM and I'll be updating it as I go along.

# I. Getting Good with PyTorch

- **PyTorch Basics**: Tensors, Operations, Autograd system for automatic differentiation, CUDA tensors for GPU acceleration.
- **Neural Networks in PyTorch**: Using `torch.nn`, defining layers, forward pass, loss functions, and optimizers.
- **Working with Data**: Datasets, DataLoaders, data preprocessing, and augmentation techniques.
- **Model Training and Validation**: Batching, training loops, validation, overfitting, underfitting, and regularization techniques.
- **Saving and Loading Models**: Checkpoints, saving best models, and model inference.

# II. Learning about to LLMs
- **Architecture of Transformer Models**: Attention mechanisms, multi-head attention, positional encoding, feed-forward networks.
- **Pre-trained Models Overview**: GPT (Generative Pre-trained Transformer), BERT (Bidirectional Encoder Representations from Transformers), and variants (RoBERTa, T5, etc.).
- **Tokenization and Embeddings**: WordPiece, SentencePiece, BPE (Byte Pair Encoding), contextual embeddings.
- **Language Modeling**: Unsupervised learning, predicting the next word, understanding context.
- **Evaluation Metrics**: Perplexity, BLEU score, ROUGE, F1 score, accuracy, precision, recall.

# III. Mathematical Foundations

Foundational and advanced mathematical concepts that underpin the workings of Large Language Models (LLMs), especially those based on the Transformer architecture.

1. **Linear Algebra**:
    - **Vectors and Matrices**: Understanding the basic building blocks of neural networks, including operations like addition, multiplication, and transformation.
    - **Eigenvalues and Eigenvectors**: Importance in understanding how neural networks learn and how data can be transformed.
    - **Special Matrices**: Identity matrices, diagonal matrices, and their properties relevant to neural network optimizations.

2. **Calculus**:
    - **Derivatives and Gradients**: Essential for understanding the backpropagation algorithm and how neural networks learn by minimizing loss functions.
    - **Partial Derivatives and Chain Rule**: Crucial for training models using gradient descent and for understanding the autograd system in PyTorch.

3. **Probability and Statistics**:
    - **Probability Theory**: Basics including probability distributions, expectations, variance, and covariance.
    - **Bayesian Methods**: Understanding how prior knowledge is updated with evidence using Bayes' theorem is critical for some models and applications.
    - **Statistical Measures**: Mean, median, mode, standard deviation, and their importance in data preprocessing and understanding model performance.

4. **Optimization Theory**:
    - **Convex Optimization**: While not all problems in deep learning are convex, the concepts are foundational and help in understanding various optimization algorithms.
    - **Gradient Descent and Variants**: Deep dive into how gradient descent works, including its variants like stochastic gradient descent (SGD), Adam, etc.
    - **Loss Functions**: Understanding different types of loss functions and their applications in training neural networks.

5. **Information Theory**:
    - **Entropy and Information Content**: Basic concepts of information theory that underpin many models' objective functions.
    - **Cross-Entropy and KL Divergence**: Important for understanding the loss functions used in training classification models and generative models.

6. **Discrete Mathematics**:
    - **Graph Theory**: Useful for understanding attention mechanisms and data structures that represent relationships and interactions in data.
    - **Combinatorics**: Foundations for understanding the complexity of model architectures and for tasks such as sequence generation.

7. **Numerical Methods**:
    - **Numerical Stability and Conditioning**: Important for training models, especially to understand and mitigate issues like vanishing or exploding gradients.
    - **Matrix Decompositions**: Techniques such as singular value decomposition (SVD) and QR decomposition, which are useful for certain optimization problems and understanding deep learning models.


# IV. Fine-Tuning and Optimising LLMs
- **Fine-Tuning Techniques**: Transfer learning, learning rate adjustment, layer freezing/unfreezing, gradual unfreezing.
- **Optimization Algorithms**: Adam, RMSprop, SGD, learning rate schedulers.
- **Regularization and Generalization**: Dropout, weight decay, batch normalization, early stopping.
- **Efficiency and Scalability**: Mixed precision training, model parallelism, data parallelism, distributed training.
- **Model Size Reduction**: Quantization, pruning, knowledge distillation.

# V. RAG: Retrieval-Augmented Generation
- **Introduction to RAG**: Concept, architecture, comparison with traditional LLMs.
- **Retrieval Mechanisms**: Dense Vector Retrieval, BM25, using external knowledge bases.
- **Integrating RAG with LLMs**: Fine-tuning RAG models, customizing retrieval components.
- **Applications of RAG**: Question answering, fact checking, content generation with external references.
- **Challenges and Solutions**: Handling out-of-date knowledge, bias in retrieved documents, improving retrieval relevance.

# VI. Developing real-world Applications with LLMs
- **Integrating LLMs into Applications**: API development, deploying models with Flask/Django for web applications, mobile app integration.
- **User Interface and Experience**: Chatbots, virtual assistants, generating human-like text, handling user inputs.
- **Security and Scalability**: Authentication, authorization, load balancing, caching.
- **Monitoring and Maintenance**: Logging, error handling, continuous integration and deployment (CI/CD) pipelines.
- **Case Studies and Project Ideas**: Content generation, summarization, translation, sentiment analysis, automated customer service.

---

# My Study Notes

Most of my notes will be in the form of notebooks, and I will link them in each section.
I will also write a short summary of the key points I've learned in each section.

### Before getting started

At the moment I prefer to use PyCharmPro as my dev environment. The benefits are venv- and notebook support and full IDE support (with CoPilot). 
If you want to run any of my code, you need to set up and activate a virtual environment and install the required packages with:
```bash
pip install -r requirements.txt
```
Alternatively follow these installation guides
* https://pytorch.org/get-started
* https://jupyter.org/install

## PyTorch

I am a software engineer and already know how to code. But I am new to the PyTorch library and want to get familiar and fluent writing code with it before I dive deeper into LLMs. 
If you don't know how to program, I would recommend to take at least a short introductory course into Python before continuing. 

### Tensors

* [001-pytorch-tensors.ipynb](https://github.com/florianbuetow/llm-fasttrack/blob/main/notebooks/001-pytorch-tensors.ipynb): Learning to work with tensors in PyTorch

### Summary
- PyTorch tensor is an n-dimensional array that is the same as a NumPy array or TensorFlow tensor. 
- A rank 0 tensor as a scalar, a rank 1 tensor as a vector, and a rank 2 tensor as a matrix.
- Tensors of different dimension and with different sizes can be created from Python lists, NumPy arrays, or random values.
- The size and dimension of a tensor can be accessed using the `size()` and `dim()` methods.

```python
import torch
my_tensor = torch.randn((2, 3, 4), dtype=torch.float)
print("The dtype of my tensor a is:", my_tensor.dtype)
print("The size of my tensor a is:", my_tensor.size())
print("The shape of my tensor a is:", my_tensor.shape)
print("The dims of my tensor a is:", my_tensor.dim())
print("The dims of my tensor a is:", my_tensor.ndim)
print("The number of elements in my tensor is:", my_tensor.numel())
print("My tensor is stored on the GPU:", my_tensor.is_cuda)
print("My tensor is stored on device:", my_tensor.device)
```
PyTorch offers different data types, choosing the right one is important because it will influence the memory usage and the performance. 
Some PyTorch methods have requirements on the datatype of the tensor as well.

| Data Type                | dtype                       | CPU tensor         | GPU tensor              |
|--------------------------|-----------------------------|--------------------|-------------------------|
| 32-bit floating point    | torch.float32/torch.float   | torch.FloatTensor  | torch.cuda.FloatTensor  |
| 64-bit floating point    | torch.float64/torch.double  | torch.DoubleTensor | torch.cuda.DoubleTensor |
| 8-bit integer (signed)   | torch.int16                 | torch.ShortTensor  | torch.cuda.ShortTensor  |
| boolean                  | torch.bool                  | torch.BoolTensor   | torch.cuda.BoolTensor   |


---

# Terms and Concepts

| Keyword                      | Explanation                                                                                                                                                                                                                                                                                                                                                                                        | Links                                                                                                       |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **Temperature**              | affects the randomness of the model's output by scaling the logits before applying softmax, influencing the model's "creativity" or certainty in its predictions. Lower temperatures lead to more deterministic outputs, while higher temperatures increase diversity and creativity. | [Peter Chng](https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/) 
| **Top P (Nucleus Sampling)** |selects a subset of likely outcomes by ensuring the cumulative probability exceeds a threshold p, allowing for adaptive and context-sensitive text generation. This method focuses on covering a certain amount of probability mass. | [Peter Chng](https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/) |
| **Top K**                    |limits the selection pool to the K most probable next words, reducing randomness by excluding less likely predictions from consideration. This method normalizes the probabilities of the top K tokens to sample the next token.                        | [Peter Chng](https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/) |
| **Q (Query)**                |represents the input tokens being compared against key-value pairs in attention mechanisms, facilitating the model's focus on different parts of the input sequence for predictions.                                                                |
| **K (Key)**                  |represents the tokens used to compute the amount of attention that input tokens should pay to the corresponding values, crucial for determining focus areas in the model's attention mechanism.                                                       |
| **V (Value)**                |is the content that is being attended to, enriched through the attention mechanism with information from the key, indicating the actual information the model focuses on during processing.                                                         |
| **Embeddings**               |are high-dimensional representations of tokens that capture semantic meanings, allowing models to process words or tokens by encapsulating both syntactic and semantic information.                                                                |
| **Tokenizers**               |are tools that segment text into manageable pieces for processing by models, with different algorithms affecting model performance and output quality.                                                                                             |
| **Rankers**                  |are algorithms used to order documents or predict their relevance to a query, influencing the selection of next words or sentences based on certain criteria in NLP applications.                                                                     |


# Reading List 

In this section I keep track of all the articles, papers, and tutorials I want to go through.

Up Next:

- [How I got into deep learning](https://www.vikas.sh/post/how-i-got-into-deep-learning): Vikas Paruchuri's journey into deep learning and AI.

Queue:
- [Neural Networks - From the ground up](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi): YouTube series from 3Blue1Brown 
- [But what is a GPT? Visual intro to transformers](https://www.youtube.com/watch?v=wjZofJX0v4M): Chapter 5, Deep Learning
- [Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=308s): Chapter 6, Deep Learning
- [Token Selection Strategies: Top-k, Top-p, and Temperature](https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/): by Peter Chng
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY): by Andrej Karpathy
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/): A visual guide to the GPT-2 model architecture.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/): A visual guide to the Transformer model architecture.
- [The Annotated GPT-2](https://amaarora.github.io/posts/2020-02-18-annotatedGPT2.html): A detailed explanation of the GPT-2 model architecture.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html): A detailed explanation of the Transformer model architecture.
- [The Transformer: Attention is All You Need](https://arxiv.org/abs/1706.03762): The original paper that introduced the Transformer model.
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): The original paper that introduced the GPT-2 model.
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165): The original paper that introduced the GPT-3 model.
- [Hugging Face Transformers](https://huggingface.co/transformers/): A library of pre-trained models for NLP tasks.
- [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM): A curated list of resources for Large Language Models.
