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

# The Curriculum

Is constantly evolving as I learn more. Here's what I'm focusing on right now:

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

# III. Fine-Tuning and Optimising LLMs
- **Fine-Tuning Techniques**: Transfer learning, learning rate adjustment, layer freezing/unfreezing, gradual unfreezing.
- **Optimization Algorithms**: Adam, RMSprop, SGD, learning rate schedulers.
- **Regularization and Generalization**: Dropout, weight decay, batch normalization, early stopping.
- **Efficiency and Scalability**: Mixed precision training, model parallelism, data parallelism, distributed training.
- **Model Size Reduction**: Quantization, pruning, knowledge distillation.

# IV. Developing real-world Applications with LLMs
- **Integrating LLMs into Applications**: API development, deploying models with Flask/Django for web applications, mobile app integration.
- **User Interface and Experience**: Chatbots, virtual assistants, generating human-like text, handling user inputs.
- **Security and Scalability**: Authentication, authorization, load balancing, caching.
- **Monitoring and Maintenance**: Logging, error handling, continuous integration and deployment (CI/CD) pipelines.
- **Case Studies and Project Ideas**: Content generation, summarization, translation, sentiment analysis, automated customer service.

# V. RAG: Retrieval-Augmented Generation
- **Introduction to RAG**: Concept, architecture, comparison with traditional LLMs.
- **Retrieval Mechanisms**: Dense Vector Retrieval, BM25, using external knowledge bases.
- **Integrating RAG with LLMs**: Fine-tuning RAG models, customizing retrieval components.
- **Applications of RAG**: Question answering, fact checking, content generation with external references.
- **Challenges and Solutions**: Handling out-of-date knowledge, bias in retrieved documents, improving retrieval relevance.

---

# My Study Notes

### Setup notes

I use PyCharm as by development environment because it automatically creates a virtual environment for my code and offers the support of a classic IDE. The pro version even comes with Jupyter Notebook support.

The installation of pytorch and jupyter notebook inside a virtual environment is straight forward.
https://pytorch.org/get-started
https://jupyter.org/install

Or you can use the following commands to install the required packages listed in the requirements.txt file.
```bash
pip install -r requirements.txt
```
Just make sure to perform the installation while inside a virtual environment.

## PyTorch

### Tensors

- PyTorch tensor is an n-dimensional array that is the same as a NumPy array or TensorFlow tensor. 
- A rank 0 tensor as a scalar, a rank 1 tensor as a vector, and a rank 2 tensor as a matrix.


---

# Reading List 

In this section I keep track of all the articles, papers, and tutorials I want to go through.

- [Neural Networks - From the ground up](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi): YouTube series from 3Blue1Brown 
- [But what is a GPT? Visual intro to transformers](https://www.youtube.com/watch?v=wjZofJX0v4M): Chapter 5, Deep Learning
- [Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=308s): Chapter 6, Deep Learning 
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/): A visual guide to the GPT-2 model architecture.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/): A visual guide to the Transformer model architecture.
- [The Annotated GPT-2](https://amaarora.github.io/posts/2020-02-18-annotatedGPT2.html): A detailed explanation of the GPT-2 model architecture.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html): A detailed explanation of the Transformer model architecture.
- [The Transformer: Attention is All You Need](https://arxiv.org/abs/1706.03762): The original paper that introduced the Transformer model.
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): The original paper that introduced the GPT-2 model.
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165): The original paper that introduced the GPT-3 model.
- [Hugging Face Transformers](https://huggingface.co/transformers/): A library of pre-trained models for NLP tasks.
