# Learning Diary

### 07 November 2017
  - [Bawden et al - Evaluating Discourse Phenomena in NMT](https://arxiv.org/pdf/1711.00513.pdf)
    - Analysis of the impact of inter-sentence context in terms of coreference resolution, lexical disambiguation.
    - Models:
      - Classical attentive NMT baseline
      - Concatenated src_prev-src_current -> trg
      - Concatenated src_prev-src_current -> trg_prev, trg_current
      - Multi-source multi-attention similar to our fusion model
      

### 03 November 2017
  - [Alcantara - Empirical analysis of non-linear activation functions for Deep Neural Networks in classification tasks](https://arxiv.org/pdf/1710.11272.pdf)
    - Small report that compares sigmoid, ReLU, ELU, SELU and LReLU over MNIST.
    - ReLu, Leaky ReLu, ELU and SELU activation functions all yield great results in terms of validation error and accuracy on the MNIST task, with the ELU layer overall performing better than all other models.

### 20 October 2017
  - [Lee et al - Emergent Translation in Multi-Agent Communication](https://arxiv.org/pdf/1710.06922.pdf)
    - Two agents communicate with each other in their own respective languages to solve a visual
referential task. One agent sees an **image** and describes it in **its native language** to the other agent.
The other agent is **given several images**, one of which is the same image shown to the first agent,
and has to choose the **correct image** using the description.
    - The game is played in both directions simultaneously, and the agents are jointly trained to solve this task. We only allow agents to send a sequence of discrete symbols to each other, and never a continuous vector.

### 19 October 2017
  - [Stop using word2vec (blog)](http://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/)
  - [Essence of Linear Algebra - I](https://www.youtube.com/watch?v=kjBOesZCoqc)

### 18 October 2017
  - [Eleni Vasilaki - Is Epicurus the Father of Reinforcement Learning?](https://arxiv.org/pdf/1710.04582v1.pdf)
  - [Ramachandran et al - Swish: A self-gated activation function](https://arxiv.org/pdf/1710.05941.pdf)
     - Swish is a novel activation function with the form f(x) = x · sigmoid(x).
     - Swish has the properties of one-sided boundedness at zero, smoothness, and non-monotonicity, which may play a role in
the observed efficacy of Swish and similar activation functions.
     - Our experiments used models and hyperparameters that were designed for ReLU and just replaced the ReLU activation function with Swish; even this simple, suboptimal procedure resulted in Swish consistently outperforming ReLU and other activation functions. We expect additional gains to be made when these models and hyperparameters are specifically designed with Swish in mind. The simplicity of Swish and its similarity to ReLU means that replacing ReLUs in any network is just a simple one line code change.

### 17 September 2017
  - [Wang et al - Translating Phrases in Neural Machine Translation (EMNLP17)](https://arxiv.org/pdf/1708.01980)
    - Word-by-word generation in NMT makes it difficult to translate multi-word phrases.
      - Zhang and Zong 2016 - Bridging neural machine translation and bilingual dictionaries.
      - Stahlberg et al 2016 - Syntactically guided neural machine translation.
    - Auxiliary phrase memory to store target phrases in symbolic form:
      - Written by SMT using NMT decoding information.
      - NMT decoder scores phrases and performs a word/or/token selection.
    - Encoder enriched with syntactic chunk information.
    - Architecture: A sigmoid-scalar MLP (balancer) defining the trade-off between word/phrase generators.
      - Linguistically improved source representation is used.
      - Phrase memory updated each decoding step using NMT info, phrases are scored with SMT.
      - Separate probabilities are computed during beam search, fused with balancer.
      - If a phrase has been selected, the decoder updates its decoding state by consuming the words in it.
    - Results: +0.5 BLEU with memory, +1.0 with memory+chunking but **no results** for only chunking.
    - Related work section is pretty rich.

  - [Weng et al - Neural Machine Translation with Word Predictions (EMNLP17)](https://arxiv.org/pdf/1708.01771.pdf)
    - Hidden states to predict the target vocabulary, ensure better encoder and decoder representations.
    - Britz et al. 2017 find that the decoder initialization does not affect the translation performance. Here authors argue that initial state is important and neglected and supervises it additionally.
    - The claims in 4.1 about error propagation is wrong for attentive networks.
    - WPE: Puts a softmax estimation over the initial hidden state s_0 of the decoder, applies attention over encoder states.
    - WPD: Predict the remaining/untranslated words from each hidden decoder state.
    - All approaches can be further used to constrain target vocab during inference.
    - Consistent improvements achieved based on the results.
    - Q: Why not train from scratch and instead use a pre-trained baseNMT?
    - Nice test about precision/recall of target words.
  
### 23 August 2017
 - [Serdyuk et al - Twin Networks: Using the Future as a Regularizer](https://arxiv.org/abs/1708.06742)
   - Classical sequence-to-sequence extended with an additional right-to-left RNN decoder which generates the reversed target tokens.
   - The loss is extended with this new sequence loss as well as a ~L2 distance between the hidden states of these two RNNs where the idea is to learn an affine transform between the forward decoder and the backward decoder.
   - The ~L2 distance loss is back-propagated only to the forward decoder.
   - %12 relative improvement in CER for a speech recognition task.
   - (**Comment**: Similar to L2R-R2L works in NLP/MT)

### 21 August 2017
 - [Bello et al - Neural Optimizer Search with Reinforcement Learning](http://proceedings.mlr.press/v70/bello17a/bello17a.pdf)
   - Learns custom update rules using a controller RNN and RL.
   - Transfers well to other tasks like NMT, i.e. an optimizer with an update rule of `g * exp(sign(g)*sign(ravg(g)))` improves upon ADAM.

### 9 August 2017
 - [Zhao et al - Learning Sleep Stages from Radio Signals: A Conditional Adversarial Architecture](http://sleep.csail.mit.edu/files/rfsleep-paper.pdf) \[[WEB](http://sleep.csail.mit.edu)\]
   - Reflected RF spectrograms encoded with CNN + RNN
   - 3-way adversarial setup pushing the system to learn a source-invariant representation

### 4 August 2017

 - [Jan Niehues, Eunah Cho - Exploiting Linguistic Resources for Neural Machine Translation Using Multi-task Learning](https://arxiv.org/pdf/1708.00993.pdf)
   - Multi-task setup with POS and NE recognition as additional tasks.
   - Each mini-batch represents only a single task.
   - Sequence length information is used in decoding for POS tagging task.
   - Multi-task setup seems to improve MT baseline.

 - [Merity et al - Revisiting Activation Regularization for Language RNNs](https://arxiv.org/pdf/1708.01009.pdf)
   - L2 activation regularization (AR): The output of the RNN (h_t) is regularized
     - \alpha L2(drop_mask * h_t)
   - Temporal activation regularization (TAR): Penalizes any large changes in h_t between timesteps.
     - \beta L2(h_t - h_{t+1})
   - Perplexity is improved consistently.

### 3 August 2017

 - [Xing Shi, Kevin Knight - Speeding Up Neural Machine Translation Decoding by Shrinking Run-time Vocabulary](http://aclanthology.coli.uni-saarland.de/pdf/P/P17/P17-2091.pdf)
 - [Sennrich et al - The University of Edinburgh's Neural MT Systems for WMT17](http://arxiv.org/pdf/1708.00726.pdf)
 - [Niehues et al - Analyzing Neural MT Search and Model Performance](http://arxiv.org/pdf/1708.00563.pdf)
 - [van der Wees et al - Dynamic Data Selection for Neural Machine Translation (EMNLP17)](http://arxiv.org/pdf/1708.00712.pdf) :thumbsup:
   - Relevance scores by Axelrod cross-entropy difference, epoch sampling w.r.t those scores
 
 ### 2 August 2017
 
 - [Miceli Barone et al - Regularization techniques for fine-tuning in neural machine translation](http://arxiv.org/pdf/1707.09920.pdf)
 
### 1 August 2017

 - [Levin et al - Machine Translation at Booking.com: Journey and Lessons Learned](http://arxiv.org/pdf/1707.07911.pdf)
