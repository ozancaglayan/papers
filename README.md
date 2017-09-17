# Reading Diary

### 17 September 2017
 - [Weng et al - Neural Machine Translation with Word Predictions](https://arxiv.org/pdf/1708.01771.pdf)
   - 

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
 - [van der Wees et al - Dynamic Data Selection for Neural Machine Translation](http://arxiv.org/pdf/1708.00712.pdf) :thumbsup:
   - Relevance scores by Axelrod cross-entropy difference, epoch sampling w.r.t those scores
 
 ### 2 August 2017
 
 - [Miceli Barone et al - Regularization techniques for fine-tuning in neural machine translation](http://arxiv.org/pdf/1707.09920.pdf)
 
### 1 August 2017

 - [Levin et al - Machine Translation at Booking.com: Journey and Lessons Learned](http://arxiv.org/pdf/1707.07911.pdf)
