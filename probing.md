# Probing - Literature Review


## Bird’s Eye: Probing for Linguistic Graph Structures with a Simple Information-Theoretic Approach (ACL-IJCNLP 2021)
* New information theoretic probe for **linguistic graphs**. Instead of depending on clasifier accuracy,this estimates the mutual information between the linguitic graph embedded in a continuous space and the contextualized word repreenttqations - called Bird's eye.
* Information theoretic approach is more reliable than training a clf as it can be arguedif the clf is probing or trying to solve the task.
* Also propose to probe localized linguistic infromation in the graph using perturbation analysis - called Worm's eye.
![](https://i.imgur.com/wCYvPPH.png)

*In this paper we propose a general information-theoretic probe method, which is capable of prob- ing for linguistic graph structures and avoids the randomness of training a model. In the experiments, we use our probe method to show the extent to which syntax trees and semantic graphs are encoded in pretrained BERT models. Further, we perform a simple perturbation analysis to show that with small modifications, the probe can also be used to probe for specific linguistic sub-structures. There are some limitations of our probe. First, a graph embedding is used, and some structure infor- mation could be lost in this process. We provide simple ways to test this. Second, training a MI estimation model is difficult. Future work can con- sider building on our framework by exploring better graph embedding and MI estimation techniques.*


## Probing Toxic Content in Large Pre-Trained Language Models (acl-ijcnlp 2021) [Relevant]
We propose a method based on <span style="color:blue">*logistic regression classifiers to probe*</span> English, French, and Arabic PTLMs and quantify the potentially harmful content that they convey with <span style="color:blue">*respect to a set of templates. The templates are prompted by a name of a social group followed by a cause-effect relation*</span>. We use PTLMs to <span style="color:blue">predict masked tokens at the end of a sentence in order to examine how likely they enable toxicity towards specific communities</span>. We shed the light on how such negative content can be triggered within unrelated and benign contexts based on evidence from a large-scale study, then we explain how to take advantage of our methodology to assess and mitigate the toxicity transmitted by PTLMs.

![](https://i.imgur.com/jD01Ask.png)
![](https://i.imgur.com/7mQc8IM.png)
![](https://i.imgur.com/uRnz6xc.png)


* We train simple toxicity classifiers based on logistic regression using available hate speech and offensive language datasets. We reduce the classification bias using a two-step approach to first, filter out examples with identity words which typically lead classifiers to predict a toxic label, then perform a second classification step on the remaining examples.

##### METHODOLOGY:
* We use a PTLM on a one token masked cloze statement which starts with the name of a social group, followed by an everyday action, and ends by a predicted reason of the action.
* We use the ATOMIC atlas of everyday common-sense reasoning based on if-then relations (Sap et al., 2019b) to create cloze statements to fill in. <span style="color:blue">We generate 378,000 English sentences, 198,300 French, and 160,552 Arabic sentences using the presented patterns.</span>
* We prompt the statements by the name of a social group and use gendered pronouns to evoke the effect of the action.
* **PROBING CLASSIFIER:** We run our probing experiments in two steps:
    * In the first step, we run the LR classifier on cloze statements which contain patterns based on different social groups and actions without using the generated content. Then, we remove all the patterns which have been classified as toxic. 
    * In the second step, we run our classifier over the full generated sentences with only patterns which were not labeled toxic. In this case, we consider the toxicity of a sentence given the newly PTLM-introduced content. Finally, we compare counts of potentially incoherent associations produced by various PTLMs in English, French, and Arabic.
* The PTLMs we use are BERT, RoBERTa, GPT-2, CamemBERT, and AraBERT.
*  In the first step, 9.55%, 83.55%, and 18.25% of the English, French, and Arabic sentences to be probed were filtered out by the toxic language classifiers. After filtering out the toxic patterns that our classifier labeled as offensive, we fed the sentences generated from the remaining patterns to be labeled by the toxic language classifiers. The overall results for three PTLMs in English and the two Arabic and French PTLMs are shown in Table 5.
![](https://i.imgur.com/dehxATn.png)

* These are then evaluated by humans.
![](https://i.imgur.com/2jMojYW.png)

* Overall results as classified by the various PTLMs:
![](https://i.imgur.com/fTan6oy.png)

* <span style="color:blue">**Frequent Content in English**</span> Present a table of top-1 predictions for different types of prompts. I dont remember what the paper did, but an idea could be to ategorize the prompts based on some pattern, like POSTag sequence, dependency structure/chain or something.

On the other hand, Ettinger (2020) introduced a series of psycholinguistic diagnosis tests to evaluate what PTLMs are not designed for, and Bender et al. (2021) thoroughly surveyed their impact in the short and long terms.

**TLDR:** They present a methodology to <span style="color:red">**probe toxic content in pre-trained language models using commonsense patterns.**</span>

## Are we there yet? Exploring clinical domain knowledge of BERT models
*BioNLP Workshop at ACL 2021*
Explore whether state-of-the-art BERT models encode sufficient domain knowledge to correctly perform domain-specific inference. Already established that BERT impls such as BioBERT are better than general BERTs for domain specific reasoning. They supplement the PTLMs domain knowledge via four methods to observe its effect: 
1. Further pretrain on medical domain corpora
2. By lexical match algos such as BM25
3. Supplementing lexical relations with dependency relations
4. Using a trained retriever model

<span style="color:red">No difference is found between knowledge supplemented classification and the baseline BERT, which is attributed to unreliable knowledge retrieval for complex domain reasoning. **We find that the errors related to domain knowledge-based reasoning, such as the knowledge of treatments administered for certain diseases, are dominant (40%) in BioBERT.**</span>

<span style="color:blue">The goal of our study is to understand whether these methods can be successfully applied for knowledge integration in the more complex setup of finding missing medical information for supporting sentence-pair inference. We explore both **implicit** and **explicit** knowledge integration, where implicit refers to indirect access to this knowledge by further language model pretraining on medical corpora, and explicit knowledge integration refers to the setup where a relevant sentence from external corpora is appended to the premise to support inference. For explicit knowledge integration, as the baseline method, we make use of the traditional best match 25 (BM25) algorithm (Robertson and Zaragoza, 2009) for finding the most relevant sentence in the medical corpora. As a modification of this method, we additionally incorporate syntactic knowledge in the retrieval step. We do so by restricting the retrieved sentence to the one that contains at least one dependency relation between premise and hypothesis medical entities. In the third setup, instead of using BM25 scores and dependency paths, we train an end-toend model to first find the most relevant text block from Wikipedia for a given instance, and then append it to the instance for classification. </span>

**Medical Language Inference**
MedNLI dataset - Premise+Hypothesis pair and labels are [Entails, Contradicts, Neutral (relationship cannot be established)].
The task is modelled as a sentence pair claification task, where the final pooled BERT [CLS] representation of the premise and the hypothesis are passed through a dene neural network to classify the label. Manual analysis of 50 incorrectly classified instances in the development set.
![](https://i.imgur.com/tP7AqCz.png)

**Medical Knowledge Augmentation:** The three forms of augmentation are:
**External Medical Corpora:** Create two corpora - Wikimed (medical subset of Wikipedia) and Medbook (contents of a popular medical book).
**Implicit Knowledge Integration:** Continue BioBERT pretraining on the above two corpora
**Explicit Knowledge Integration:** Three types of integration are:
![](https://i.imgur.com/ZOqdIhD.png)

1. Lexical Retrieval: Construct a query from premise and hypothesi by retaining only lemmas that are a part of medical entities, and use BM25 to rank documents in the external corpora. Prune the top 25 retrieved sentences if they do not mention at least one premise and one hypothesis entity lemma. The highest ranking sentence retrieved in this manner is then appended3 to the premise before classification. If none of the sentences satisfy either the constraint or the threshold score, then the use of explicit knowledge is skipped.
2. Lexical and Syntactic Retrieval: Add constraint o that retrieved sentence is abot both premise and hypothesis. From top documents, restrict the set to sentences that have a dependency relation between premise and hypothesis lemma.
3. Joint Retrieval and classification model: Shortlisting just one sentence from a lot is challenging. So train an end-to-end retriever whose weights are updated alongwith the classification. Retriever is pretrained in an invere cloze task setup on Wikipedia. Retrieval is performed by computing a weighted dot product between the pooled BERT [CLS] embeddings of the query and the text block.  In an end-to-end setup, the retriever module first returns the k5 most similar blocks of text given a BERT-encoded premise and hypothesis pair, in the same manner as described earlier.  We then encode these inputs with BERT to obtain k different [CLS] representations. All of these k [CLS] representations are then individually used for classification by adding a dense layer on the top in the finetuning phase. In this manner, we obtain k different outputs for a given instance. We then aggregate these k outputs together by retaining the most frequent output among the k options.
![](https://i.imgur.com/p9CXjbV.png)
We find that none of the explored methods provide better access to medical information for domain knowledge-based reasoning, although the desired factual information is present in these external corpora. Retrieval of relevant information for language inference demands a delicate balance between selecting a sentence that provides sufficient supporting information related to the given topic and instance to improve inference, and yet that is neither redundant nor superfluous.

## Exploring the Role of BERT Token Representations to Explain Sentence Probing Results (EMNLP 2021) [Relevant]
Usually diagnodtic classifiers are trained on repreentation obtained from different layers of BERT, and clf accuracy is interpreted as the ability to encode that corresponding linguistic property. However, token representations have not been studied yet. We provide a more in-dept analysis of the representation space of BERT in search for distinct and meaningful subspaces that explain the reasons behind these probing results. 
<span style="color:red">Based on a set of probing tasks and with the help of attribution methods we show that BERT tends to encode meaningful knowledge in specific token representations (which are often ignored in standard classification setups), allowing the model to detect syntactic and semantic abnormalities, and to distinctively separate grammatical number and tense subspaces.</span>
<span style="color:blue">Recent works in probing language models demonstrate that initial layers are responsible for encoding low-level linguistic information, such as part of speech and positional information, whereas intermediate layers are better at syntactic phenomena, such as syntactic tree depth or subject-verb agreement, while in general semantic information is spread across the entire model (Lin et al., 2019; Peters et al., 2018; Liu et al., 2019a; Hewitt and Manning, 2019; Tenney et al., 2019).</span>

BERT usually encodes the knowledge required for addressing these tasks within specific token representations, particularly at higher layers. For instance, we found that sentence-ending tokens (e.g., “SEP” and “.”) are mostly responsible for carrying positional information through layers, or when the input sequence undergoes a re-ordering the alteration is captured by specific token representations, e.g., by the swapped tokens or the coordinator between swapped clauses. <span style="color:blue">Also, we observed that the ##s token is mainly responsible for encoding noun number and verb tense information, and that BERT clearly distinguishes the two usages of the token in higher layer representations.</span>

##### *Representation Subspaces*
<span style="color:blue">Hewitt and Manning (2019) </span> show that there exists a linear subspace that approximately encodes all syntactic distances. Chi et al. (2020) show that similar subspaces exist for languages other than English in mBERT and these are shared among languages to some extent. 

Semantic Subspaces: <span style="color:blue">Wiedemann et al. (2019)</span> show that BERT places contextualized representations of polysemous words in different regions of the embedding space, thereby capturing different sense distinctions. <span style="color:blue">Reif et al. (2019) </span> find that there exists a linear transformation under which distances between word embeddings correspond to their sense-level relationships.



## What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties (ACL 2018) [Relevant]
Downtream tasks often utilise the sentence representations, but their complexity make it difficult to infer what information is present in these representations. Information probed by this paper:
1. Surface Information: Surface properties of text. E.g.:
    a. SentenceLength: Make 6 equal-width bins and do a 6-way classification task for each sentence.
    b. WordContent: Pick 1k words in mid-frequency range (2k-3k). Sample equal number of sentence that cotnain only one of these words and do a 1k-way classification task.
2. Syntactic Information: Test for syntactic properties:
    a. BigramShift: Clf has to distinguish between sentences where a bigram is inverted.
    b. TreeDepth: Check for the hierarchical structure of the sentence, i.e group sentences by the length of the longest path from the root to any leaf. For their dataset, the depth varies from 5 to 12, and the probing task is a 8-way classifier.
    c. TopConstitutentTask: 20-way classification task. Given a sentence, find the top most constituent sequence.
3. Semantic Information: Information about the emantic content:
    a. Tense: Tense of the main clause verb
    b. SubjNum: The number of the subject of the main clause
    c. ObjectNumber: The number of direct objects in the main clause.
    d. SOMO (Semantic Odd Man OUt): Randomly replace a noun or a verb with anoter noun or verb. The bigram frequences of the replaced word is ensured to be comparable (on a log scale) before and after the replacement.
    e. CoordinationInversion: Invert the order of the clause in half of the sentences, and make the classifer predict these.
    
The following sentence embedding models were inspected:
1. BiLSTM-last/max: $h_t$ iss the concat of the forward and the backward LSTM. For T words in the sequence, we have {$h_1$, .. , $h_T} vectors sequence. To create a fixed dimension vector, either we take the maximum value in each dimension, or take the last hidden state $h_T$, i.e. $h_T$.
2. Gated ConvNet: 
![](https://i.imgur.com/rpYtdEh.png)
![](https://i.imgur.com/G5P0rzK.png)


## **Probing Biomedical Embeddings from Language Models** [Relevant]
Venue: NAACL-HLT 2019 Workshop on Evaluating Vector Space Representations for NLP (RepEval)
* Use pretrained LMs as fixed feature extractors, and restrict the downtream task models to not have additional sequence modelling layers.
* Compare BERT, ELMO, BioBERT, BioElMo. Finetuned BioBERT is better than BioElmo in biomedical NER and NLI tasks, BioElmo outperforms BioBERT as a fixed feature extractor in probing. 
* Better encoding of entity-type and relational information is the reason for superior performance.
* Tenney et al (2018) (What do you learn from context? Probing for sentence structure in contextualized word representations) extend token-level probing to span-level probing.
* NER Downstream Task: For BioELMo, following Lample et al.(2016), we use the contextualized embeddings and a character-based CNN for word representations, which are fed to a biLSTM, followed by a conditional random field (CRF) (Lafferty et al., 2001) layer for tagging. For BioBERT, we use the single sentence tagging setting described in Devlin et al. (2018), where the final hidden states of each token are trained to classify its NER label.
* NLI Downstream Task: Final hidden state of the first token CLS are trained to classify the NLI label for the sentence pair.
* **PROBING TASK:** NER Probing Task and the NLI Probing Task: 
![](https://i.imgur.com/fJR6Wdd.png)
![](https://i.imgur.com/4vLfDMJ.png)
![](https://i.imgur.com/zPLBd6Q.png)

![](https://i.imgur.com/TWfdeVP.png)

<span style="color:red">***Control Setting***</span>
![](https://i.imgur.com/vDfEYi3.png)

Results on the NER and NLI task:


|                 NER                  |                 NLI                  |
|:------------------------------------:|:------------------------------------:|
| ![](https://i.imgur.com/uNEbX8t.png) | ![](https://i.imgur.com/4ihslzF.png) |


In biomedical literature, the acronym ER has multiple meanings: out of the 124 mentions we found in 20K recent PubMed abstracts, 47 refer to the gene “estrogen receptor”, 70 refer to the organelle “endoplasmic reticulum” and 4 refer to the “emergency room” in hospital.
![](https://i.imgur.com/6geG1PQ.png)
<span style="color:blue">BioELMo better clusters entities from the same types together. Unlike ELMo/BioELMo, Whether the ER mention is inside parentheses doesn’t affect BERT/BioBERT representations. It can be explained by encoder difference between ELMo and BERT: For ELMo, to predict ‘)’ in forward LM, representations of token ‘ER’ inside the parentheses need to encode parentheses information due to the recurrent nature of LSTM. For BERT, to predict ‘)’ in masked LM, the masked token can attend to ‘(’ without interacting with ‘ER’ representations, so BERT ‘ER’ embedding does’t need to encode parentheses information.</span>

**Nearest Neighbourhood Analysis**
We evaluate relation representations from different embeddings by nearest neighbor (NN) analysis: For each distributed relation representation (Eq. 1) of these token pairs, we calculated the ***proportions of its five nearest neighbors*** that belong to the same relation type. 
![](https://i.imgur.com/OXfia0E.png)


## Does My Representation CaptureX? Probe-Ably (ACL-IJCNLP Demo 2021) (Mendeley)
A probing framework deigned for PyTorch which supports and automates the application of best probing practices to the user's inputs.

Some existing works discuss these best practices: Controlling and varying model complexity and structure, including randomized control tasks, incorporating more informative metrics such as selectivity and minimum description length.

Their main contributions are:
1. Configure and run probing experiments on different representations and auxiliary tasks in parallely.
2. Automatically generate contorl tasks for the probing, allowing the computation of inter-model metrics such as selectivity.
3. Extend the uite of probes with new model without the need to change the core probing pipeline.
4. Customize, implement, and adopt novel evaluation metrics for the experiment.

**Probing Pipeline**
1. Data Processing: Creating the control task by randomly assigning labels to the examples in the auxiliary task. Exhaustive hyperparameter selection for the right interpretation of the probing results, as well as the coverage of the configuration space.
2. Training Probes: Train a set of probe models $\phi$. If *n* is the number of representations to be probed, *m* be the number of aux tasks, *z* be the number of probe models, and *k* the number of selected hyperparam configurations for each probe, then total cardinality of $\phi$ = *n* * *m* * *z* * *k*.
3. Evaluation Metrics: Most common ones are *selectivity* and *accuracy*. These are plotted against the probe complexity and are used to compare trends in the performance of the different representations.

**Available Models:** Linear models (y = Wx + b) and MLP.
Complexity measure of the models:
1. Linear model ($\hat{y}$ = $Wx + b$): Nuclear norm of the matrix $W$ represented as $||W||_*$ is considered the approximate measure of complexity. The rationale is that the nuclear norm approximates the rank of the transformation matrix. A weighted nuclear norm is also included in the loss function and thus regulated in the training loop.
![](https://i.imgur.com/JYe26r2.png)

2. MLP: Number of parameters is used as a naive estimation of the mode complexity. 

**Available Metrics**
1. Intra-Model Metrics: Individual model results and losses such as cross-entropy loss and accuracy. Can be used for training, model-selection, and reporting purposes.
2. Inter-Model Metrics: Assessing the reliability of a probe's result is the selectivity metric. For a fixed probe architecture and hyperparameter config, the aux task accuracy is compared with the control task accuracy. Other metrics that can be included are **Minimum description length** or **Pareto Hypervolume** which incorporate the results of multiple models or training runs.

**Guidelines for Interpreting Results**
Regions of low selectivity indicate a less tustworthy aux task accuracy result. However, selectivity needs to be taken care of with increase in accuracy with model complexity. High accuracy on random control task means that the probe is expressive enough.
Each probe architecture imposes a structural assumption. For example, linear probes may only attain a high accuracy if the representation-target relationship is linear. We recommend that these assumptions/probe model choices be guided by prior visualizations and hypothesized relationships.
Stick to comparing representations of the same sizes. Lower-dimensional representations may reach their maximum accuracy at lower probe complexity values; as such they may give the "appearance" of superior probe accuracy scores to larger representations.

## Designing and Interpreting Probes with Control Tasks (Mendeley)
![](https://i.imgur.com/avbTsvo.png)
![](https://i.imgur.com/lKWQIWp.png)

A good probe (one that provides insights into the linguistic properties of a representation) should be what we call *selective*, achieving high linguistic task accuracy and low control task accuracy (Figure 2).

![](https://i.imgur.com/4etlXII.png)
![](https://i.imgur.com/NZvRVbP.png)
![](https://i.imgur.com/k51HJDm.png)


## On Lack of Robustness Interpretability of Neural Text Classifiers (Findings of ACL-IJCNLP 2021) [Relevant]
They assess the robustness of interpretations of neural text classifiers, specifically, those based on pretrained Transformer encoders, using two randomization tests. The first compares the interpretations of two models that are identical except for their initializations. The second measures whether the interpretations differ between a model with trained parameters and a model with random parameters. Both tests show surprising deviations from expected behavior.
<span style="color:red">The existing methods of *feature interpretability* possess *high fidelty*, i.e. removing features marked important by the interpretability method from the input indeed leads to significant change in the model output as expected. [Relevant works to read: Atanasova
et al., 2020; Lundberg and Lee, 2017].</span>

Model interpretability has different aspects: local (e.g. Lundberg and Lee, 2017) vs. global (e.g. Tan et al., 2018), feature-based (e.g. Lundberg and Lee, 2017) vs. concept-based (e.g. Kim et al., 2018) vs. hidden representation- based (Li et al., 2016). See Gilpin et al. (2018); Guidotti et al. (2018) for an overview.
Closest to ours is the work of Adebayo et al. (2018), which is based on checking the saliency maps of randomly initialized image classification models. However, in contrast to Adebayo et al., we consider text classification. Moreover, while the analysis of Adebayo et al. is largely based on visual inspection, we extend it by considering automatically quantifiable measures. We also extend the analysis to non-gradient based methods (SHAP). 

To understand the robustness of feature attributions, the following two tests are conducted:
1. **Different Initialization Test:** Given an input, it compares the feature attributions between two models that are identical in every aspect—that is, trained with same architecture, with same data, and same learning schedule—except for their randomly chosen initial parameters. 
2. **Untrained Model Test:** This test is similar to the test of Adebayo et al. (2018). Given an input, it compares the feature attributions generated on a fully trained model with those on a randomly initialized untrained model. 

*The results suggest that: (i) Interpretability methods fail the different initializations test. In other words, two functionally equivalent models lead to different ranking of feature attributions; (ii) Interpretability methods fail the untrained model test, i.e., the fidelity of the interpretability method on an untrained model is better than that of random feature attributions. *  


**Experimental Setup:**
Four datasets: FPB (Financial Phrase Bank), SST2 (Stanford Sentiment Treebank 2), IMDB Reviews, and Bios (Classify the profession of a person from biography). Four models are considered: BERT, RoBERTa, DistilBERT, and DistilRoBERTa.
<span style="color:red">We consider a mix of gradient-based and model agnostic methods. Specifically: Vanilla Saliency (VN) of Simonyan et al. (2014), SmoothGrad (SG) of Smilkov et al. (2017), Integrated Gradients (IG) of Sundararajan et al. (2017), and KernelSHAP (SHP) of Lundberg and Lee (2017). We also in- clude random feature attribution (RND) which cor- responds to each feature being assigned an attri- bution from the uniform distribution, U(0, 1). For each input feature (that is, token), the feature attribution of the gradient-based methods is a vector of the same length as the token input embedding. For scalarizing these vector scores, we use the L2-norm strategy of Arras et al. (2016) and the Input Gradient strategy of Ding et al. (2019). </span>

