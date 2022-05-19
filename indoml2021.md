# IndoML 2021 Talks

## Sameer Singh: Evaluating and Testing Natural Language Processing Models

How do you define if a new model X is better than an old model Y? One way is to pick a task such as QnA, and compare the accuray of both of these models. This is generally recorded by leaderboards, such as SQuAD. Instead, some one said it is better to compare on a bunch of tasks, one example being GLUE and Super-GLUE. However, there are a lot of issues in these models depite their high acuracies there are multiple issues. Some are:
* Shortcuts (right for wrong reasons)
* Sensitive to phrasing
* Sensitive to typos

Problems with leaderboards:
* Duplicate the bias of data collection
* Single metric is insufficinet
* Model capabilities are unknown

The truth is that we don't really know how to compare two models, or effectively evaluate which model is better for what reasons?


* Perturbation-based Testing
 (X, y) -> Transformation -> (X', y') -> Model -> $y^{\^}$
 These could be automatic or manual. 
 ***Automatic***: SEAR (ACL 2018), Triggers (EMNLP 2019), CRIAGE (NAACL 2019)
 ***Manual***: Implications (ACL 2019), Contrast Sets (FEMNLP 2020)
 A Testing Framework: Checklist (ACL 2020)

Universal Adversarial Triggers: Short phrases when added as a prefix to any input, will drop the model accuracy drastically. Models such as sentiment (or classifiers), or QA models. For example, we we can add a phrase "why how because to kill american people" when added to the paragraph, 2/3 of the answers to questions pick the answer "to kill american people".
SEARS: Paraphraical edits (which keep the meaning same). Characterizing via rules. Change the question so that the answer is inconsistent.

* Checklist Paper details. 

## Tim Baldwin: Fairness in Natural Language Processing
Some examples of Unfair AI:
* Twitter's auto-cropping algo: cchooses people with specific features, from specific racial bakgrounds etc.
* Bias in US mortgrage approvals.
* NLP technologies, such as languge identification libraries

Particular protected attributes of interest (gender, race, age, etc) vary greatly across datasets, based on data availabilty, task, and specific interest of researchers.

Talk Outline:
**1. Baseline Approaches for Mitigating Bias**
**2. Mitigating Bias with Adversarial Learning and Beyond**

* **Training Models to be Fair(er)**
It in't enough to not include the protected attributes. Sometimes, naively trained models pick up and amplify bias. If the cause is imbalance in terms of protected attributes, then prebalancing may help (Wang et al 2019). 
Another way could be to train a model to be as good as possible at predicting the target and as bad as possible to predict the hidden attributes, via adversarial learning.
One "Adversarial discriminator" per protected attribute, which predict the values of these attributes via the hidden representations. Train via min-max, i.e. minimize the cross entropy over target and maximize the CE over the protected variable. Use an iterative apporoach, where fix the discriminator and minimize the CE for target, then fix the parameters for the model, and then train the adversarial dicriminator.
E.g.: POSTagging - biLSTM model and a single FFlayer applied to final hidden representation used as an adversarial discriminator. Protected attrs were age and gender. Evaluate accuracy for both in-domain and cross-domain setting.

* **Issue 1: Training Instability**
Could be difficult to train (failure to converge). Difficult for complex models/tasks. Some engineering solutions are: add extra capacity to the adversarial discriminators, upweight teh adversarial loss term(s), ensemble the discriminator. One issue with naive ensembling of dicriminators is tha there is no constraint that they complement each other, so one fix is to introduce a orthogonality constraint on the parameters of the adversarie in a given ensemble (Bousmalis et al 2016).  This enforces diversity in adversarial discriminator.
Findings: 
    * Ensemble of discriminators > single discriminator > vanilla model. 
    * Better and more stable results with orthogonality constraints on ensemble
    

* **Issue 2: Unavailibilty of protected attributes for some users**
 The training of the adv dicriminator is decoupled from the base model, so we can just train the discriminator onthe subset of attributes that are present. Also possible to transfer the protected attributes between tasks/datasets with diffrerent target varibale from external domains.
 ![](https://i.imgur.com/h4tVgoG.png)

* **Intersectional Bias**
 The orthogonality onstraint enhanes the per-attribute performance, but it might not be good at capturing the interaction between protected attributes. So instead of looking at these attributes as independent groups, we can create Intersectional Groups (non overlapping, exhautive combinations, of groups), or Gerrymandering Interetional Groups (overlapping ubgroups defined by iterection fo any subset of private attributes). 
![](https://i.imgur.com/zDThQ43.png)
![](https://i.imgur.com/kDyqS4N.png)
![](https://i.imgur.com/9ntl2AB.png)




## Ido Dagan: Novel data for modeling textual information: decomposition, multi-text, and interaction

Two types of data in NLP:
1. Representing Language Structure
    * E.g.: Syntatic/semantic/discourse
    * Used to train models that can generate language structures (which are used as intermediate representations/features in end-task models)
2. Capturing input/output of end-task applications
QA, IE, Summarization, etc. Used to train task-specific models.

Earlier models extracted feature from the data, and trained the modls on these feature.
![](https://i.imgur.com/n6OcdWL.png)

More recently, as neural models emerged, we have the end to end paradigm, where you feed raw text to model and the model will somehow expected to represent the syntactic etc structure, and the internal model will train on the end task. The first type of data is not needed anymore. 
![](https://i.imgur.com/VlobEBL.png)

The question is now whether such implicit representations are sufficient? Can we break E2E glass ceiling via some explicit structures? Some of theri benefits are deeper emantic undertanding, interpretability, control.

Outline of talk: Novel Type of Data for NLP
1. Modelling information struture in a (single) text
2. Modelling cros-text information relations
3. Human interaction with interactive NLP applications

Continuation of the EMNLP-2021 keynote. Refer to notes taken from EMNLP keynote.


## Dragomir R. Radev: Closing the loop in natural language interfaces to databases: Parsing, dialogue, and generation

NL interfaces for structured data, e.g.: natural language query to SQL query, table language modelling, structure to text generation (QA, summarization included) forms a closed loop. 

***SPIDER***: Complex, Cross-domain text-to-QL dataset
![](https://i.imgur.com/rxCISzA.png)

***CoQL Leaderboard***: 
***GraPPa***: Grammar-Augmented Pre-Training for Table emantic Parsing (ICLR 2021)
Yin et al. (2020) and Herzig et al. (2020) pretrain LMs on over 20M text-tables extracted from the web. 
![](https://i.imgur.com/ALqFMNH.png)

![](https://i.imgur.com/9lGBvAV.png)

![](https://i.imgur.com/ViqKxwr.png)

Synchronous Context-Free Grammar (SCFG)
![](https://i.imgur.com/2T9e6Zg.png)


Two objective functions for pretraining LMs:
1. Column Contextual Semantics (CCS)
2. Turn Contextual Switch (TCS)

![](https://i.imgur.com/HPF4OmE.png)

![](https://i.imgur.com/vFCKNYI.png)

![](https://i.imgur.com/L1V5kkz.png)



<!-- ## Monojit Choudhury: Applications and techniques for Code-Mixed text generation

Why is there a need for code-mixed data? -->

## Mohit Bansal: Knowledgeable & Spatial-Temporal Vision+Language
![](https://i.imgur.com/uYHNIEj.png)
![](https://i.imgur.com/fPtuDGI.png)
![](https://i.imgur.com/vs2sJht.png)
![](https://i.imgur.com/j6B0LMS.png)
![](https://i.imgur.com/UNHqJOq.png)
![](https://i.imgur.com/79xY5Se.png)
![](https://i.imgur.com/tRhHSbz.png)


## Robert Hoehndorf: Machine learning with ontologies in biomedicine

![](https://i.imgur.com/qYNynaS.png)
While there are multiple types of datasets, the big challenge is to combine these in a meaningful way so that we can connect how the knowledge from genome, trancsriptome, metabolome etc are related.

Biomedicine is a knowledge-driven dicipline where a lot of information comes from more than 100 years of experiments. This is the reason for large biomedical knowledge bases. If we want to build models in biomedicine, which do anything complex, we also need to consider ontologies. An ontology is an explicit conceptualization of a domain, and in terms of axioms define how concepts are related. Currently, there are over 500 biomedical ontologies.
![](https://i.imgur.com/AYh55WK.png)

![](https://i.imgur.com/y8Yh4Zj.png)

* Embedding Formal Knowledge
An embedding is a map (morphism) from one mathematical structure X into another structure Y:
$f: X \rightarrow Y$, 
such that $X$ is preserved in $Y$.
![](https://i.imgur.com/cO3yVOT.png)

We need some mapping from ontologies to graphs.
![](https://i.imgur.com/rOYdJKL.png)

Once we have a graph from the ontology, then we can also get graph embedding using various algorithms such as Node2Vec or RandomWalk, TransE, or Word2Vec.

![](https://i.imgur.com/WPsspmU.png)

We started with a simple problem of relation prediction in a graph. Remove some edges, and then try to predict them via some ML algo. 
![](https://i.imgur.com/brDHK0D.png)
The results obtained via with or without reasoning didn't vary much, which was frutrating. It was expected that if we are able to use the axioms in the ontologies, then the graph in the KG well enough, then the performance should increase substantially. However, the axioms are hard to represent in a graph (see examle below).
![](https://i.imgur.com/VixWAzt.png)

So the question is, is there a way to embed these axioms? Some ideas such as Onto2Vec apply a LM to the axioms directly. But then there is no way to use the priori semantics that we know the axioms follow. For example, sigma-algebra structures such as top-concept, bottom-concept, singleton sets, etc which are used in first-order logic to define syntax and semantics between concepts and relations.
![](https://i.imgur.com/qbPwome.png)


We want to find an embedding that generates a model of the axioms in the ontology. The model means that the axioms in the ontology are true in the model.

We define an embedding for relations and classes. We map our classes to n-balls, such that points which are within the sphere/ball will be in extension of the concept. Each concept is mapped to n-ball, and we map relations as transformations between them.

Discuion of A-boxes and other ontology specific things. Summmary slide:
![](https://i.imgur.com/B0h7rJ3.png)


## Marinka Zitnik: Infusing Structure and Knowledge Into Biomedical AI

![](https://i.imgur.com/0d2A2ft.png)
We need to integrate humungou amounts of diverse datasets which is challenging. In terms of building ML models, accuracy alone is not enough. Explanations, and fairness are important. 

![](https://i.imgur.com/YNee59a.png)
![](https://i.imgur.com/RIlbzbp.png)
![](https://i.imgur.com/IhyQ1Mk.png)


1. Infusing structure into Biomedical AI
Few Shot Learning on Graphs
Existing GNNs require abundant labels, which are scarce in non CS datasets. Such as developing drugs, novel emerging diseases etc.

![](https://i.imgur.com/T6GNFYn.png)
The assumption that the graph is richly labelled(indicated by node color) and hence a GNN clasifier can be trained accurately, is wrong. Most nodes are unknown or have little knowledge as is seen in the right graph. Few-shot learning is essential in such scenarios.

**Few-Shot learning:** Only few example per label are present.
![](https://i.imgur.com/29MUKYl.png)
2-shot, 3-way: 2 classes for prediction, and for each class, we have only 2 labelledexamples/shots. The two stages are:
    i. Meta-Training: Take a model and  train it in such a way that we ask it to solve a large number of training tasks where each task is designed to be very challenging. As the model learns new tasks, it will become better and better at quickly leaning/adapting to other tasks. We want to identify a mechanism to transfer information from one training task to another so that eventually it becomes better at adaption to new tasks as it has already seen so many tasks.
    ii. Meta-Test: 
    
Few-Shot for Graphs:
One algo is G-Meta (NeurIPS)
![](https://i.imgur.com/uQ78sKM.png)
Training phase: learn different labels (represented by different shades of yellow). Train task 1 has a process to learn a better initialization for training task 2. This is then repeated for training task 3 and so on.
![](https://i.imgur.com/NiDrvKA.png)

The key idea behind quick adaption to learn new tasks is the notion of local subgraphs. In addition to learning d-dim node embeddings, we alo learn subgraph signature functions. That signatures represents the task in a very effective manner. The signature function is defined by looking at local neighbouring subgraphs around nodes that are labelled for that task.
![](https://i.imgur.com/vvtWxsb.png)

Now, why use local subgraphs for signature function? (Read TL;DR)
![](https://i.imgur.com/0tBl4Pd.png)
![](https://i.imgur.com/KGJaV9A.png)

***Next: Application in biomedical dicovery***
Rapid Therapeutic Innovation
![](https://i.imgur.com/GWa2RHT.png)
![](https://i.imgur.com/lbJruCz.png)
![](https://i.imgur.com/TFsS58D.png)


Each training task was defined for a different disease. GMeta had to predict which drugs will work or not work for that disease (approved vs unapproved drugs). The earlier training tasks were for known diseases for which some treatments were available.
![](https://i.imgur.com/EUUHnV5.png)


GMeta outperformed other network difussion based algos and classic drug repurposing algos.
![](https://i.imgur.com/myAI4P9.png)


