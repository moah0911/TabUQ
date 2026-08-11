## TABM: ADVANCING TABULAR DEEP LEARNING WITH PARAMETER-EFFICIENT ENSEMBLING

Yury Gorishniy ∗

Yandex Akim Kotelnikov HSE University, Yandex

## ABSTRACT

Deep learning architectures for supervised learning on tabular data range from simple multilayer perceptrons (MLP) to sophisticated Transformers and retrievalaugmented methods. This study highlights a major, yet so far overlooked opportunity for designing substantially better MLP-based tabular architectures. Namely, our new model TabM relies on efficient ensembling , where one TabM efficiently imitates an ensemble of MLPs and produces multiple predictions per object. Compared to a traditional deep ensemble, in TabM, the underlying implicit MLPs are trained simultaneously, and (by default) share most of their parameters, which results in significantly better performance and efficiency. Using TabM as a new baseline, we perform a large-scale evaluation of tabular DL architectures on public benchmarks in terms of both task performance and efficiency, which renders the landscape of tabular DL in a new light. Generally, we show that MLPs, including TabM, form a line of stronger and more practical models compared to attention- and retrievalbased architectures. In particular, we find that TabM demonstrates the best performance among tabular DL models. Then, we conduct an empirical analysis on the ensemble-like nature of TabM. We observe that the multiple predictions of TabM are weak individually, but powerful collectively. Overall, our work brings an impactful technique to tabular DL and advances the performance-efficiency trade-off with TabM - a simple and powerful baseline for researchers and practitioners. The code is available at: https://github.com/yandex-research/tabm .

## 1 INTRODUCTION

Supervised learning on tabular data is a ubiquitous machine learning (ML) scenario in a wide range of industrial applications. Among classic non-deep-learning methods, the state-of-the-art solution for such tasks is gradient-boosted decision trees (GBDT) (Prokhorenkova et al., 2018; Chen &amp; Guestrin, 2016; Ke et al., 2017). Deep learning (DL) models for tabular data, in turn, are reportedly improving, and the most recent works claim to perform on par or even outperform GBDT on academic benchmarks (Hollmann et al., 2023; Chen et al., 2023b;a; Gorishniy et al., 2024).

However, from the practical perspective, it is unclear if tabular DL offers any obvious go-to baselines beyond simple architectures in the spirit of a multilayer perceptron (MLP). First , the scale and consistency of performance improvements of new methods w.r.t. simple MLP-like baselines are not always explicitly analyzed in the literature. Thus, one has to infer those statistics from numerous per-dataset performance scores, which makes it hard to reason about the progress. At the same time, due to the extreme diversity of tabular datasets, consistency is an especially valuable and hard-to-achieve property for a hypothetical go-to baseline. Second , efficiency-related properties, such as training time, and especially inference throughput, sometimes receive less attention. While methods are usually equally affordable on small-to-medium datasets (e.g. &lt; 100K objects), their applicability to larger datasets remains uncertain. Third , some recent work generally suggests that the progress on academic benchmarks may not transfer that well to real-world tasks (Rubachev et al., 2024). With all the above in mind, in this work, we thoroughly evaluate existing tabular DL methods and find that non-MLP models do not yet offer a convincing replacement for MLPs.

At the same time, we identify a previously overlooked path towards more powerful, reliable, and reasonably efficient tabular DL models. In a nutshell, we find that the parameter-efficient approach to

∗ The corresponding author: yurygorishniy@gmail.com

Artem Babenko Yandex deep ensembling, where most weights are shared between ensemble members, allow one to make simple and strong tabular models out of plain MLPs. For example, MLP coupled with BatchEnsemble (Wen et al., 2020) - a long-existing method - right away outperforms popular attention-based models, such as FT-Transformer (Gorishniy et al., 2021), while being simpler and more efficient. This result alone suggests that efficient ensembling is a low-hanging fruit for tabular DL.

Our work builds on the above observations and offers TabM - a new powerful and practical model for researchers and practitioners. Drawing an informal parallel with GBDT (an ensemble of decision trees), TabM can also be viewed as a simple base model (MLP) combined with an ensembling-like technique, providing high performance and simple implementation at the same time.

Main contributions. We summarize our main contributions as follows:

1. We present TabM - a simple DL architecture for supervised learning on tabular data. TabM is based on MLP and parameter-efficient ensembling techniques closely related to BatchEnsemble (Wen et al., 2020). In particular, TabM produces M ultiple predictions per object. TabM easily competes with GBDT and outperforms prior tabular DL models, while being more efficient than attention- and retrieval-based DL architectures.
2. We provide a fresh perspective on tabular DL models in a large-scale evaluation along four dimensions: performance ranks, performance score distributions, training time, and inference throughput. One of our findings is that MLPs, including TabM, hit an appealing performanceefficiency tradeoff, which is not the case for attention- and retrieval-based models.
3. We show that the two key reasons for TabM's high performance are the collective training of the underlying implicit MLPs and the weight sharing. We also show that the multiple predictions of TabM are weak and overfitted individually, while their average is strong and generalizable.

## 2 RELATED WORK

Decision-tree-based models. Gradient-boosted decision trees (GBDT) (Chen &amp; Guestrin, 2016; Ke et al., 2017; Prokhorenkova et al., 2018) is a strong and efficient baseline for tabular tasks. GBDT is a classic machine learning model, specifically, an ensemble of decision trees. Our model TabM is a deep learning model, specifically, a parameter-efficient ensemble of MLPs.

Tabular deep learning architectures. A large number of deep learning architectures for tabular data have been proposed over the recent years. That includes attention-based architectures (Song et al., 2019; Gorishniy et al., 2021; Somepalli et al., 2021; Kossen et al., 2021; Yan et al., 2023), retrieval-augmented architectures (Somepalli et al., 2021; Kossen et al., 2021; Gorishniy et al., 2024; Ye et al., 2024), MLP-like models (Gorishniy et al., 2021; Klambauer et al., 2017; Wang et al., 2020) and others (Arik &amp; Pfister, 2020; Popov et al., 2020; Chen et al., 2023b; Marton et al., 2024; Hollmann et al., 2023). Compared to prior work, the key difference of our model TabM is its computation flow, where one TabM imitates an ensemble of MLPs by producing multiple independently trained predictions. Prior attempts to bring ensemble-like elements to tabular DL (Badirli et al., 2020; Popov et al., 2020) were not found promising (Gorishniy et al., 2021). Also, being a simple feed-forward MLP-based model, TabM is significantly more efficient than some of the prior work. Compared to attention-based models, TabM does not suffer from quadratic computational complexity w.r.t. the dataset dimensions. Compared to retrieval-based models, TabM is easily applicable to large datasets.

Improving tabular MLP-like models. Multiple recent studies achieved competitive performance with MLP-like architectures on tabular tasks by applying architectural modifications (Gorishniy et al., 2022), regularizations (Kadra et al., 2021; Jeffares et al., 2023a; Holzm¨ uller et al., 2024), custom training techniques (Bahri et al., 2021; Rubachev et al., 2022). Thus, it seems that tabular MLPs have good potential, but one has to deal with overfitting and optimization issues to reveal that potential. Our model TabM achieves high performance with MLP in a different way, namely, by using it as the base backbone in a parameter-efficient ensemble in the spirit of BatchEsnsemble (Wen et al., 2020). Our approach is orthogonal to the aforementioned training techniques and architectural advances.

Deep ensembles. In this paper, by a deep ensemble, we imply multiple DL models of the same architecture trained independently (Jeffares et al., 2023b) for the same task under different random seeds (i.e. with different initializations, training batch sequences, etc.). The prediction of a deep ensemble is the mean prediction of its members. Deep ensembles often significantly outperform single DL models of the same architecture (Fort et al., 2020) and can excel in other tasks like uncertainty estimation or out-of-distribution detection (Lakshminarayanan et al., 2017). It was observed that individual members of deep ensembles can learn to extract diverse information from the input, and the power of deep ensembles depends on this diversity (Allen-Zhu &amp; Li, 2023). The main drawback of deep ensembles is the cost and inconvenience of training and using multiple models.

Parameter-efficient deep 'ensembles'. To achieve the performance of deep ensembles at a lower cost, multiple studies proposed architectures that imitate ensembles by producing multiple predictions with one model (Lee et al., 2015; Zhang et al., 2020; Wen et al., 2020; Havasi et al., 2021; Antor´ an et al., 2020; Turkoglu et al., 2022). Such models can be viewed as 'ensembles' where the implicit ensemble members share a large amount of their weights. There are also non-architectural approaches to efficient ensembling, e.g. FGE (Garipov et al., 2018), but we do not explore them, because we are interested specifically in architectural techniques. In this paper, we highlight parameter-efficient ensembling as an impactful paradigm for tabular DL. In particular, we describe two simple variations of BatchEnsemble (Wen et al., 2020) that are highly effective for tabular MLPs. One variation uses a more efficient parametrization, and another one uses an improved initialization.

## 3 TABM

In this section, we present TabM - a Tab ular DL model that makes M ultiple predictions.

## 3.1 PRELIMINARIES

Notation. We consider classification and regression tasks on tabular data. x and y denote the features and a label, respectively, of one object from a given dataset. A machine learning model takes x as input and produces ˆ y as a prediction of y . N ∈ N and d ∈ N respectively denote the 'depth' (e.g. the number of blocks) and 'width' (e.g. the size of the latent representation) of a given neural network. d y ∈ N is the output representation size (e.g. d y = 1 for regression tasks, and d y equals the number of classes for classification tasks).

Datasets. Our benchmark consists of 46 publicly available datasets used in prior work, including Grinsztajn et al. (2022); Gorishniy et al. (2024); Rubachev et al. (2024). The main properties of our benchmark are summarized in Table 1, and more details are provided in Appendix C.

Table 1: The overview of our benchmark. The 'Split type' property is explained in the text.

|   #Datasets | Train size   | Train size   | Train size   | Train size   | #Features   | #Features   | #Features   | #Features   | Task type   | Task type   | Split type   | Split type   |
|-------------|--------------|--------------|--------------|--------------|-------------|-------------|-------------|-------------|-------------|-------------|--------------|--------------|
|             | Min.         | Q50          | Mean         | Max.         | Min.        | Q50         | Mean        | Max.        | #Regr.      | #Classif.   | Random       | Domain-aware |
|          46 | 1.8K         | 12K          | 76K          | 723K         | 3           | 20          | 108         | 986         | 28          | 18          | 37           | 9            |

Domain-aware splits. We pay extra attention to datasets with what we call 'domain-aware' splits, including the eight datasets from the TabReD benchmark (Rubachev et al., 2024) and the Microsoft dataset (Qin &amp; Liu, 2013). For these datasets, their original real-world splits are available, e.g. time-aware splits as in TabReD. Such datasets were shown to be challenging for some methods because they naturally exhibit a certain degree of distribution shift between training and test parts (Rubachev et al., 2024). The random splits of the remaining 37 datasets are inherited from prior work.

Experiment setup. We use the setup from Gorishniy et al. (2024), and describe it in detail in subsection D.2. Most importantly, on each dataset, a given model undergoes hyperparameter tuning on the validation set, then the tuned model is trained from scratch under multiple random seeds, and the test metric averaged over the random seeds becomes the final score of the model on the dataset.

Metrics. We use RMSE (the root mean square error) for regression tasks, and accuracy or ROC-AUC for classification tasks depending on the dataset source. See subsection D.3 for details.

Also, throughout the paper, we often use the relative performance of models w.r.t. MLP as the key metric. This metric gives a unified perspective on all tasks and allows reasoning about the scale of improvements w.r.t. to a simple baseline (MLP). Formally, on a given dataset, the metric is defined as ( score baseline -1 ) · 100% , where 'score' is the metric of a given model, and 'baseline' is the metric of MLP. In this computation, for regression tasks, we convert the raw metrics from RMSE to R 2 to better align the scales of classification and regression metrics.

## 3.2 A QUICK INTRODUCTION TO BATCHENSEMBLE.

For a given architecture, let's consider any linear layer l in it: l ( x ) = Wx + b , where x ∈ R d 1 , W ∈ R d 2 × d 1 , b ∈ R d 2 . To simplify the notation, let d 1 = d 2 = d . In a traditional deep ensemble, the i -th member has its own set of weights W i , b i for this linear layer: l i ( x i ) = W i x i + b i , where x i is the object representation within the i -th member. By contrast, in BatchEnsemble, this linear layer is either (1) fully shared between all members, or (2) mostly shared: l i ( x i ) = s i ⊙ ( W ( r i ⊙ x i )) + b i , where ⊙ is the elementwise multiplication, W ∈ R d × d is shared between all members, and r i , s i , b i ∈ R d are not shared between the members. This is equivalent to defining the i -th weight matrix as W i = W ⊙ ( s i r T i ) . To ensure diversity of the ensemble members, r i and s i of all members are initialized randomly with ± 1 . All other layers are fully shared between the members of BatchEnsemble.

The described parametrization allows packing all ensemble members in one model that simultaneously takes k objects as input, and applies all k implicit members in parallel, without explicitly materializing each member. This is achieved by replacing one or more linear layers of the original neural network with their BatchEnsemble versions: l BE ( X ) = (( X ⊙ R ) W ) ⊙ S + B , where X ∈ R k × d stores k object representations (one per member), and R,S,B ∈ R d store the non-shared weights ( r i , s i , b i ) of the members, as shown at the lower left part of Figure 1.

Terminology. In this paper, we call r i , s i , b i , R , S and B adapters , and the implicit members of parameter-efficient emsembles (e.g. BatchEnsemble) implicit submodels or simply submodels .

Overhead to the model size. With BatchEnsemble, adding a new ensemble member means adding only one row to each of the matrices R , S , and B , which results in 3 d new parameters per layer. For typical values of d , this is a negligible overhead to the original layer size d 2 + d .

Overhead to the runtime. Thanks to the modern hardware, the large number of shared weights and the parallel execution of the k forward passes, the runtime overhead of BatchEnsemble can be (significantly) lower than × k (Wen et al., 2020). Intuitively, if the original workload underutilizes the hardware, there are more chances to pay less than × k overhead.

## 3.3 ARCHITECTURE

TabM is one model representing an ensemble of k MLPs. Contrary to conventional deep ensembles, in TabM, the k MLPs are trained in parallel and share most of their weights by default, which leads to better performance and efficiency. We present multiple variants of TabM that differ in their weight-sharing strategies, where TabM and TabMmini are the most effective variants, and TabMpacked is a conceptually important variant potentially useful in some cases. We obtain our models in several steps, starting from essential baselines. We always use the ensemble size k = 32 and analyze this hyperparameter in subsection 5.3. In subsection A.1, we explain that using MLP as the base model is crucial because of its excellent efficiency.

MLP. We define MLP as a sequence of N simple blocks followed by a linear prediction head: MLP ( x ) = Linear ( Block N ( . . . ( Block 1 ( x ))) , where Block i ( x ) = Dropout ( ReLU ( Linear (( x ))) .

MLP × k = MLP + Deep Ensemble. We denote the traditional deep ensemble of k independently trained MLPs as MLP × k . To clarify, this means tuning hyperparameters of one MLP, then independently training k tuned MLPs under different random seeds, and then averaging their predictions. The performance of MLP × k is reported in Figure 2. Notably, the results are already better and more stable than those of FT-Transformer (Gorishniy et al., 2021) - the popular attention-based baseline.

Although the described approach is a somewhat default way to implement an ensemble, it is not optimized for the task performance of the ensemble. First, for each of the k MLPs, the training is stopped based on the individual validation score, which is optimal for each individual MLP, but can be suboptimal for their ensemble. Second, the hyperparameters are also tuned for one MLP without knowing about the subsequent ensembling. All TabM variants are free from these issues.

TabMpacked = MLP + Packed-Ensemble. As the first step towards better and more efficient ensembles of MLPs, we implement k MLPs as one large model using Packed-Ensemble (Laurent et al., 2023). This results in TabMpacked illustrated in Figure 1. As an architecture, TabMpacked is equivalent to MLP × k and stores k independent MLPs without any weight sharing. However, the critical difference is that TabM processes k inputs in parallel, which means that one training step of TabM consists of k parallel training steps of the individual MLPs. This allows monitoring the performance of the ensemble during the training and stopping the training when it is optimal for the whole ensemble, not for individual MLPs. As a consequence, this also allows tuning hyperparameters for TabMpacked as for one model. As shown in Figure 2, TabMpacked delivers significantly better performance compared to MLP × k . Efficiency-wise, for typical depth and width of MLPs, the runtime overhead of TabMpacked is noticeably less than × k due to the parallel execution of the k forward passes on the modern hardware. Nevertheless, the × k overhead of TabMpacked to the model size motivates further exploration.

TabMnaive = MLP + BatchEnsemble. To reduce the size of TabMpacked, we now turn to weight sharing between the MLPs, and naively apply BatchEnsemble (Wen et al., 2020) instead of PackedEnsemble, as described in subsection 3.2. This gives us TabMnaive- a preliminary version of TabM. In fact, the architecture (but not the initialization) of TabMnaive is already equivalent to that of TabM, so Figure 1 is applicable. Interestingly, Figure 2 reports higher performance of TabMnaive compared to TabMpacked. Thus, constraining the ensemble with weight sharing turns out to be a highly effective regularization on tabular tasks. The alternatives to BatchEnsemble are discussed in subsection A.1.

<!-- image -->

Figure 1: (Upper left) Ahigh-level illustration of TabM. One TabM represents an ensemble of k MLPs processing k inputs in parallel. The remaining parts of the figure are three different parametrizations of the k MLP backbones. (Upper right) TabMpacked consists of k fully independent MLPs. (Lower left) TabM is obtained by injecting three non-shared adapters R , S , B in each of the N linear layers of one MLP ( ∗ the initialization differs from Wen et al. (2020)). (Lower right) TabMmini is obtained by keeping only the very first adapter R of TabM and removing the remaining 3 N -1 adapters. (Details) Input transformations such as one-hot-encoding or feature embeddings (Gorishniy et al., 2022) are omitted for simplicity. Drop denotes dropout (Srivastava et al., 2014).

<!-- image -->

±

±

±

±

Figure 2: The performance of models described in subsection 3.3 on 46 datasets from Table 1; plus several baselines on the left. For a given model, one dot on a jitter plot describes the performance score on one of the 46 datasets. The box plots describe the percentiles of the jitter plots: the boxes describe the 25th, 50th, and 75th percentiles, and the whiskers describe the 10th and 90th percentiles. Outliers are clipped. The numbers at the bottom are the mean and standard deviations over the jitter plots. For each model, hyperparameters are tuned. 'Model × k ' denotes an ensemble of k models.

TabMmini = MLP + MiniEnsemble. By construction, the just discussed TabMnaive (illustrated as 'TabM' in Figure 1) has 3 N adapters: R , S and B in each of the N blocks. Let's consider the very first adapter, i.e. the first adapter R in the first linear layer. Informally, its role can be described as mapping the k inputs living in the same representation space to k different representation spaces before the tabular features are mixed with @ W for the first time. A simple experiment reveals that this adapter is critical. First, we remove it from TabMnaive and keep the remaining 3 N -1 adapters untouched, which gives us TabMbad with worse performance, as shown in Figure 2. Then, we do the opposite: we keep only the very first adapter of TabMnaive and remove the remaining 3 N -1 adapters, which gives us TabMmini -the minimal version of TabM. TabMmini is illustrated in Figure 1, where we call the described approach 'MiniEnsemble'. Figure 2 shows that TabMmini performs even slightly better than TabMnaive, despite having only one adapter instead of 3 N adapters.

TabM = MLP + BatchEnsemble + Better initialization. The just obtained results motivate the next step. We go back to the architecture of TabMnaive with all 3 N adapters, but initialize all multiplicative adapters R and S , except for the very first one, deterministically with 1 . As such, at initialization, the deterministically initialized adapters have no effect, and the model behaves like TabMmini, but these adapters are free to add more expressivity during training. This gives us TabM, illustrated in Figure 1. Figure 2 shows that TabM is the best variation so far.

Hyperparameters. Compared to MLP, the only new hyperparameter of TabM is k -the number of implicit submodels. We heuristically set k = 32 and do not tune this value. We analyze the influence of k in subsection 5.3. We also share additional observations on the learning rate in subsection A.3.

Limitations and practical considerations are commented in subsection A.4.

## 3.4 IMPORTANT PRACTICAL MODIFICATIONS OF TABM

- ♠ ∼ Shared training batches . Recall that the order of training objects usually varies between ensemble members, because of the random shuffling with different seeds. For TabM, in terms of Figure 1, that corresponds to X storing k different training objects { x i } k i =1 . We observed that reusing the training batches between the TabM's submodels results in only minor performance loss on average (depending on a dataset), as illustrated with TabM ♠ in Figure 2. In practice, due to the simpler implementation and better efficiency, sharing training batches can be a reasonable starting point.
- † ∼ Non-linear feature embeddings . In Figure 2, TabM † mini denotes TabMmini with non-linear feature embeddings from (Gorishniy et al., 2022), which demonstrates the high utility of feature embeddings for TabM. Specifically, we use a slightly modified version of the piecewise-linear embeddings (see subsection D.8 for details).
- × N ∼ Deep ensemble . In Figure 2, TabM †× 5 mini denotes an ensemble of five independent TabM † mini models, showing that TabM itself can benefit from the conventional deep ensembling.

## 3.5 SUMMARY

The story behind TabM shows that technical details of how to construct and train an ensemble have a major impact on task performance. Most importantly, we highlight simultaneous training of the (implicit) ensemble members and weight sharing between them. The former is responsible for the ensemble-aware stopping of the training, and the latter apparently serves as a form of regularization.

## 4 EVALUATING TABULAR DEEP LEARNING ARCHITECTURES

Now, we perform an empirical comparison of many tabular models, including TabM.

## 4.1 BASELINES

In the main text, we use the following baselines: MLP (defined in subsection 3.3), FT-Transformer denoted as 'FT-T' (the attention-based model from Gorishniy et al. (2021)), SAINT (the attentionand retrieval-based model from Somepalli et al. (2021)), T2G-Former denoted as 'T2G' (the attentionbased model from Yan et al. (2023)), ExcelFormer denoted as 'Excel' (the attention-based model from Chen et al. (2023a)), TabR (the retrieval-based model from Gorishniy et al. (2024)), ModernNCA

denoted as 'MNCA' (the retrieval-based model from Ye et al. (2024)) and GBDT, including XGBoost (Chen &amp; Guestrin, 2016), LightGBM (Ke et al., 2017) and CatBoost (Prokhorenkova et al., 2018).

The models with non-linear feature embeddings from Gorishniy et al. (2022) are marked with † or ‡ depending on the embedding type (see subsection D.8 for details on feature embeddings):

- MLP † and TabM † mini use a modified version of the piecewise-linear embeddings.
- TabR ‡ , MNCA ‡ , and MLP ‡ (also known as MLP-PLR) use various periodic embeddings.

More baselines are evaluated in Appendix B. Implementation details are provided in Appendix D.

## 4.2 TASK PERFORMANCE

We evaluate all models following the protocol announced in subsection 3.1 and report the results in Figure 3 (see also the critical difference diagram in Figure 9). We make the following observations:

1. The performance ranks render TabM as the top-tier DL model.
2. The middle and right parts of Figure 3 provide a fresh perspective on the per-dataset metrics. TabM holds its leadership among the DL models. Meanwhile, many DL methods turn out to be no better or even worse than MLP on a non-negligible number of datasets, which shows them as less reliable solutions, and changes the ranking, especially on the domain-aware splits (right).
3. One important characteristic of a model is the weakest part of its performance profile (e.g. the 10th or 25th percentiles in the middle plot) since it shows how reliable the model is on 'inconvenient' datasets. From that perspective, MLP † seems to be a decent practical option between the plain MLP and TabM, especially given its simplicity and efficiency compared to retrieval-based alternatives, such as TabR and ModernNCA.

Summary. TabM confidently demonstrates the best performance among tabular DL models, and can serve as a reliable go-to DL baseline. This is not the case for attention- and retrieval-based models. Overall, MLP-like models, including TabM, form a representative set of tabular DL baselines.

<!-- image -->

↓

↑

↑

Figure 3: The task performance of tabular models on the 46 datasets from Table 1. (Left) The mean and standard deviations of the performance ranks over all datasets summarize the head-to-head comparison between the models on all datasets. (Middle &amp; Right) The relative performance w.r.t. the plain multilayer perceptron (MLP) allows reasoning about the scale and consistency of improvements over this simple baseline. One dot of a jitter plot corresponds to the performance of a model on one of the 46 datasets. The box plots visualize the 10th, 25th, 50th, 75th, and 90th percentiles of the jitter plots. Outliers are clipped. The separation in random and domain-aware dataset splits is explained in subsection 3.1. ( ∗ Evaluated under the common protocol without data augmentations)

## 4.3 EFFICIENCY

Now, we evaluate tabular models in terms of training and inference efficiency, which becomes a serious reality check for some of the methods. We benchmark exactly those hyperparameter configurations of models that are presented in Figure 3 (see subsection B.3 for the motivation).

TabM †∗ mini &amp;TabM †♠∗ mini . Additionally, in this section, we mark with the asterisk ( ∗ ) the versions of TabM enhanced with two efficiency-related plugins available out-of-the-box in PyTorch (Paszke et al., 2019): the automatic mixed precision (AMP) and torch.compile (Ansel et al., 2024). The purpose of those TabM variants is to showcase the potential of the modern hardware and software for a powerful tabular DL model, and they should not be directly compared to other DL models. However, the implementation simplicity of TabM plays an important role, because it facilitates the seamless integration of the aforementioned PyTorch plugins.

Training time. We focus on training times on larger datasets, because on small datasets, all methods become almost equally affordable, regardless of the formal relative difference. Nevertheless, in Figure 10, we provide measurements on small datasets as well. The left side of Figure 4 reveals that TabM offers practical training times. By contrast, the long training times of attention- and retrieval-based models become one more limitation of these methods.

Inference throughput. The right side of Figure 4 tells essentially the same story as the left side. In subsection B.3, we also report the inference throughput on GPU with large batch sizes.

Applicability to large datasets. In Table 2, we report metrics on two large datasets. As expected, attention- and retrieval-based models struggle, yielding extremely long training times, or being simply inapplicable without additional effort. See subsection D.4 for implementation details.

Parameter count. Most tabular networks are overall compact. This, in particular, applies to TabM, because its size is by design comparable to MLP. We report model sizes in subsection B.3.

Summary. Simple MLPs are the fastest DL models, with TabM being the runner-up. The attentionand retrieval-based models are significantly slower. Overall, MLP-like models, including TabM, form a representative set of practical and accessible tabular DL baselines.

<!-- image -->

Time (

↓

)

↑

Figure 4: Training times (left) and inference throughput (right) of the models from Figure 3. One dot represents a measurement on one dataset. TabM †∗ mini is the optimized TabM † mini (see subsection 4.3).

Table 2: RMSE (upper rows) and training times (lower rows) on two large datasets. The best values are in bold. The meaning of model colors follows Figure 3.

|              | #Objects   |   #Features | XGBoost       | MLP           | TabM †♠∗ mini    | TabM † mini       | FT - T            | TabR   |
|--------------|------------|-------------|---------------|---------------|------------------|-------------------|-------------------|--------|
| Maps Routing | 6 . 5 M    |         986 | 0 . 1601 28 m | 0 . 1592 15 m | 0 . 1583 2 h     | 0 . 1582 13 . 5 h | 0 . 1594 45 . 5 h | OOM    |
| Weather      | 13 M       |         103 | 1 . 4234 10 m | 1 . 4842 15 m | 1 . 4090 1 . 3 h | 1 . 4112 3 . 3 h  | 1 . 4409 13 . 5 h | OOM    |

## 5 ANALYSIS

## 5.1 PERFORMANCE AND TRAINING DYNAMICS OF THE INDIVIDUAL SUBMODELS

Recall that the prediction of TabM is defined as the mean prediction of its k implicit submodels that share most of their weights. In this section, we take a closer look at these submodels.

For the next experiment, we intentionally simplify the setup as described in detail in subsection D.5. Most importantly, all models have the same depth 3 and width 512 , and are trained without early stopping, i.e. the training goes beyond the optimal epochs. We use TabMmini from Figure 1 with k = 32 denoted as TabM k =32 mini . We use TabM k =1 mini (i.e. essentially one plain MLP) as a natural baseline for the submodels of TabM k =32 mini , because each of the 32 submodels has the architecture of TabM k =1 mini .

We visualize the training profiles on four diverse datasets (two classification and two regression problems of different sizes) in Figure 5. As a reminder, the mean of the k individual losses is what is explicitly optimized during the training of TabMmini, the loss of the collective mean prediction corresponds to how TabMmini makes predictions on inference, and TabM k =1 mini is just a baseline .

Figure 5: The training profiles of TabM k =32 mini and TabM k =1 mini as described in subsection 5.1. (Upper) The training curves. k = 32[ i ] represents the mean i ndividual loss over the 32 submodels. (Lower) Same as the first row, but in the train-test coordinates: each dot represents some epoch from the first row, and the training generally goes from left to right. This allows reasoning about overfitting by comparing test loss values for a given train loss value.

<!-- image -->

In the upper row of Figure 5, the collective mean prediction of the submodels is superior to their individual predictions in terms of both training and test losses. After the initial epochs, the training loss of the baseline MLP is lower than that of the collective and individual predictions.

In the lower row of Figure 5, we see a stark contrast between the individual and collective performance of the submodels. Compared to the baseline MLP, the submodels look overfitted individually, while their collective prediction exhibits substantially better generalization. This result is strict evidence of a non-trivial diversity of the submodels: without that, their collective test performance would be similar to their individual test performance. Additionally, we report the performance of the B est submodel of TabM across many datasets under the name TabM [ B ] in Figure 6. As such, individually, even the best submodel of TabM is no better than a simple MLP.

Summary. TabM draws its power from the collective prediction of weak, but diverse submodels.

## 5.2 SELECTING SUBMODELS AFTER TRAINING

The design of TabM allows selecting only a subset of submodels after training based on any criteria, simply by pruning extra prediction heads and the corresponding rows of the adapter matrices. To showcase this mechanics, after the training, we G reedily construct a subset of TabM's submodels with the best collective performance on the validation set, and denote this 'pruned' TabM as TabM [ G ] . The performance reported in Figure 6 shows that TabM [ G ] is slightly behind the vanilla TabM. On average over 46 datasets, the greedy submodel selection results in 8 . 8 ± 6 . 6 submodels out of the initial k = 32 , which can result in faster inference. See subsection D.6 for implementation details.

<!-- image -->

±

±

-

±

±

Figure 6: The performance on the 46 datasets from Table 1. TabM [ B ] and TabM [ G ] are described in subsection 5.1 and subsection 5.2.

<!-- image -->

Figure 7: The average performance of TabM with n layers of the width d across 17 datasets as a function of k .

## 5.3 HOW DOES THE PERFORMANCE OF TABM DEPEND ON k ?

To answer the question in the title, we consider TabM with n layers of the size d and different values of k , and report the average performance over multiple datasets in Figure 7 (the implementation details are provided in subsection D.7). The solid curves correspond to n = 3 , and the dark green curves correspond to d = 512 . Our main observations are as follows. First, it seems that the 'larger' TabM is (i.e. when n and d increase), the more submodels it can accommodate effectively. For example, note how the solid curves corresponding to different d diverge at k = 2 and k = 4 . Second, too high values of k can be detrimental. Perhaps, weight sharing limits the number of submodels that can productively 'coexist' in one network, despite the presence of non-shared adapters. Third , too narrow ( d = 64 ) or too shallow ( n = 1 ) configurations of TabM can lead to suboptimal performance, at least in the scope of middle-to-large datasets considered in this work.

## 5.4 PARAMETER-EFFICIENT ENSEMBLING REDUCES THE NUMBER OF DEAD NEURONS

Here, we show empirically that the design of TabM naturally leads to higher utilization of the backbone's weights. Even without technical definitions, this sounds intuitive, since TabM has to implement k (diverse) computations using the amount of weights close to that of one MLP.

Let's consider TabMmini as illustrated in Figure 1. By design, each of the shared neurons of TabMmini is used k times per forward pass, where 'neuron' refers to the combination of the linear transformation and the subsequent nonlinearity (e.g. ReLU). By contrast, in plain MLP (or in TabMmini with k = 1 ), each neuron is used only once per forward pass. Thus, technically, a neuron in TabMmini has more chances to be activated, which overall may lead to lower portion of dead neurons in TabMmini compared to MLP (a dead neuron is a neuron that never activates, and thus has no impact on the prediction). Using the experiment setup from subsection 5.1, we compute the portion of dead neurons in TabMmini using its best validation checkpoint. On average across 46 datasets, for k = 1 and k = 32 , we get 0 . 29 ± 0 . 17 and 0 . 14 ± 0 . 09 portion of dead neurons, respectively, which is in line with the described intuition. Technically, on a given dataset, this metric is computed as the percentage of neurons that never activate on a fixed set of 2048 training objects.

## 6 CONCLUSION &amp; FUTURE WORK

In this work, we have demonstrated that tabular multilayer perceptrons (MLPs) greatly benefit from parameter-efficient ensembling. Using this insight, we have developed TabM - a simple MLPbased model with state-of-the-art performance. In a large-scale comparison with many tabular DL models, we have demonstrated that TabM is ready to serve as a new powerful and efficient tabular DL baseline. Along the way, we highlighted the important technical details behind TabM and discussed the individual performance of the implicit submodels underlying TabM.

One idea for future work is to bring the power of (parameter-)efficient ensembles to other, non-tabular, domains with optimization-related challenges and, ideally, lightweight base models. Another idea is to evaluate TabM for uncertainty estimation and out-of-distribution (OOD) detection on tabular data, which is inspired by works like Lakshminarayanan et al. (2017).

Reproducibility statement. The code is provided in the following repository: link. It contains the implementation of TabM, hyperparameter tuning scripts, evaluation scripts, configuration files with hyperparameters (the TOML files in the exp/ directory), and the report files with the main metrics (the JSON files in the exp/ directory). In the paper, the model is described in section 3, and the implementation details are provided in Appendix D.

## REFERENCES

- Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A next-generation hyperparameter optimization framework. In KDD , 2019. 18
- Zeyuan Allen-Zhu and Yuanzhi Li. Towards understanding ensemble, knowledge distillation and self-distillation in deep learning. In ICLR , 2023. 3
- Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will Constable, Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind, Brian Hirsh, Sherlock Huang, Kshiteej Kalambarkar, Laurent Kirsch, Michael Lazos, Mario Lezcano, Yanbo Liang, Jason Liang, Yinghai Lu, C. K. Luk, Bert Maher, Yunjie Pan, Christian Puhrsch, Matthias Reso, Mark Saroufim, Marcos Yukio Siraichi, Helen Suk, Shunting Zhang, Michael Suo, Phil Tillet, Xu Zhao, Eikan Wang, Keren Zhou, Richard Zou, Xiaodong Wang, Ajit Mathews, William Wen, Gregory Chanan, Peng Wu, and Soumith Chintala. Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation. In ASPLOS , 2024. 8
- Javier Antor´ an, James Urquhart Allingham, and Jos´ e Miguel Hern´ andez-Lobato. Depth uncertainty in neural networks. In NeurIPS , 2020. 3
- Sercan O. Arik and Tomas Pfister. TabNet: Attentive interpretable tabular learning. arXiv , 1908.07442v5, 2020. 2
- Sarkhan Badirli, Xuanqing Liu, Zhengming Xing, Avradeep Bhowmik, Khoa Doan, and Sathiya S. Keerthi. Gradient boosting neural networks: GrowNet. arXiv , 2002.07971v2, 2020. 2
- Dara Bahri, Heinrich Jiang, Yi Tay, and Donald Metzler. SCARF: Self-supervised contrastive learning using random feature corruption. In ICLR , 2021. 2
- Jintai Chen, Jiahuan Yan, Danny Ziyi Chen, and Jian Wu. ExcelFormer: A neural network surpassing gbdts on tabular data. arXiv , 2301.02819v1, 2023a. 1, 6, 20, 24, 25
- Kuan-Yu Chen, Ping-Han Chiang, Hsin-Rung Chou, Ting-Wei Chen, and Tien-Hao Chang. Trompt: Towards a better deep neural network for tabular data. In ICML , 2023b. 1, 2, 15
- Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In SIGKDD , 2016. 1, 2, 7
- Stanislav Fort, Huiyi Hu, and Balaji Lakshminarayanan. Deep ensembles: A loss landscape perspective. arXiv , 1912.02757v2, 2020. 2
- Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P. Vetrov, and Andrew Gordon Wilson. Loss surfaces, mode connectivity, and fast ensembling of dnns. In NeurIPS , 2018. 3
- Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. Revisiting deep learning models for tabular data. In NeurIPS , 2021. 2, 4, 6, 15, 23, 25
- Yury Gorishniy, Ivan Rubachev, and Artem Babenko. On embeddings for numerical features in tabular deep learning. In NeurIPS , 2022. 2, 5, 6, 7, 14, 15, 20, 21, 23
- Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, and Artem Babenko. TabR: Tabular deep learning meets nearest neighbors. In ICLR , 2024. 1, 2, 3, 6, 17, 18, 19, 20, 22, 23, 24, 25

- Leo Grinsztajn, Edouard Oyallon, and Gael Varoquaux. Why do tree-based models still outperform deep learning on typical tabular data? In NeurIPS, the 'Datasets and Benchmarks' track , 2022. 3, 17, 18, 23, 24, 28
- Marton Havasi, Rodolphe Jenatton, Stanislav Fort, Jeremiah Zhe Liu, Jasper Snoek, Balaji Lakshminarayanan, Andrew Mingbo Dai, and Dustin Tran. Training independent subnetworks for robust prediction. In ICLR , 2021. 3, 14
- Noah Hollmann, Samuel M¨ uller, Katharina Eggensperger, and Frank Hutter. TabPFN: A transformer that solves small tabular classification problems in a second. In ICLR , 2023. 1, 2, 15
- David Holzm¨ uller, L´ eo Grinsztajn, and Ingo Steinwart. Better by default: Strong pre-tuned mlps and boosted trees on tabular data. arXiv , 2407.04491v1, 2024. 2
- Alan Jeffares, Tennison Liu, Jonathan Crabb´ e, Fergus Imrie, and Mihaela van der Schaar. TANGOS: Regularizing tabular neural networks through gradient orthogonalization and specialization. In ICLR , 2023a. 2
- Alan Jeffares, Tennison Liu, Jonathan Crabb´ e, and Mihaela van der Schaar. Joint training of deep ensembles fails due to learner collusion. In NeurIPS , 2023b. 2
- Arlind Kadra, Marius Lindauer, Frank Hutter, and Josif Grabocka. Well-tuned simple nets excel on tabular datasets. In NeurIPS , 2021. 2
- Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. LightGBM: A highly efficient gradient boosting decision tree. Advances in neural information processing systems , 30:3146-3154, 2017. 1, 2, 7
- Myung Jun Kim, L´ eo Grinsztajn, and Ga¨ el Varoquaux. CARTE: pretraining and transfer for tabular learning. arXiv , abs/2402.16785v1, 2024. 16
- G¨ unter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter. Self-normalizing neural networks. In NIPS , 2017. 2, 15
- Jannik Kossen, Neil Band, Clare Lyle, Aidan N. Gomez, Tom Rainforth, and Yarin Gal. Self-attention between datapoints: Going beyond individual input-output pairs in deep learning. In NeurIPS , 2021. 2
- Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In NeurIPS , 2017. 3, 10, 14
- Olivier Laurent, Adrien Lafage, Enzo Tartaglione, Geoffrey Daniel, Jean-Marc Martinez, Andrei Bursuc, and Gianni Franchi. Packed ensembles for efficient uncertainty estimation. In ICLR , 2023. 4
- Stefan Lee, Senthil Purushwalkam, Michael Cogswell, David J. Crandall, and Dhruv Batra. Why M heads are better than one: Training a diverse ensemble of deep networks. arXiv , abs/1511.06314, 2015. 3, 14
- Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR , 2019. 18
- Sascha Marton, Stefan L¨ udtke, Christian Bartelt, and Heiner Stuckenschmidt. GRANDE: Gradientbased decision tree ensembles for tabular data. In ICLR , 2024. 2
- Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas K¨ opf, Edward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, highperformance deep learning library. In NeurIPS , 2019. 8
- F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research , 12:2825-2830, 2011. 18

- Sergei Popov, Stanislav Morozov, and Artem Babenko. Neural oblivious decision ensembles for deep learning on tabular data. In ICLR , 2020. 2
- Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. CatBoost: unbiased boosting with categorical features. In NeurIPS , 2018. 1, 2, 7
- Tao Qin and Tie-Yan Liu. Introducing LETOR 4.0 datasets. arXiv , 1306.2597v1, 2013. 3
- Ivan Rubachev, Artem Alekberov, Yury Gorishniy, and Artem Babenko. Revisiting pretraining objectives for tabular deep learning. arXiv , 2207.03208v1, 2022. 2
- Ivan Rubachev, Nikolay Kartashev, Yury Gorishniy, and Artem Babenko. TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks. arXiv , 2406.19380v4, 2024. 1, 3, 17, 18, 19, 20, 22, 23, 25, 35
- Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild, C. Bayan Bruss, and Tom Goldstein. SAINT: improved neural networks for tabular data via row attention and contrastive pre-training. arXiv , 2106.01342v1, 2021. 2, 6, 25
- Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and Jian Tang. Autoint: Automatic feature interaction learning via self-attentive neural networks. In CIKM , 2019. 2, 15, 26
- Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research , 15(1):1929-1958, 2014. 5
- Ilya O. Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, and Alexey Dosovitskiy. Mlp-mixer: An all-mlp architecture for vision. In NeurIPS , 2021. 15
- Mehmet Ozgur Turkoglu, Alexander Becker, H¨ useyin Anil G¨ und¨ uz, Mina Rezaei, Bernd Bischl, Rodrigo Caye Daudt, Stefano D'Aronco, Jan D. Wegner, and Konrad Schindler. Film-ensemble: Probabilistic deep learning via feature-wise linear modulation. In NeurIPS 2022 , 2022. 3, 14, 15
- Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed H. Chi. Dcn v2: Improved deep &amp; cross network and practical lessons for web-scale learning to rank systems. arXiv , 2008.13535v2, 2020. 2, 15
- Yeming Wen, Dustin Tran, and Jimmy Ba. Batchensemble: an alternative approach to efficient ensemble and lifelong learning. In ICLR , 2020. 2, 3, 4, 5, 14, 15
- Jiahuan Yan, Jintai Chen, Yixuan Wu, Danny Z. Chen, and Jian Wu. T2G-FORMER: organizing tabular features into relation graphs promotes heterogeneous feature interaction. In AAAI , 2023. 2, 6, 23, 24
- Han-Jia Ye, Huai-Hong Yin, and De-Chuan Zhan. Modern neighborhood components analysis: A deep tabular baseline two decades later. arXiv , 2407.03257v1, 2024. 2, 7, 20, 23
- Shaofeng Zhang, Meng Liu, and Junchi Yan. The diversified ensemble neural network. In NeurIPS , 2020. 3

## A ADDITIONAL DISCUSSION ON TABM

## A.1 MOTIVATION

Why BatchEnsemble? Among relatively ease-to-use 'efficient ensembling' methods, beyond BatchEnsemble, there are examples such as dropout ensembles (Lakshminarayanan et al., 2017), naive multi-head architectures, TreeNet (Lee et al., 2015). However, in the literature, they were consistently outperformed by more advanced methods, including BatchEnsemble (Wen et al., 2020), MIMO (Havasi et al., 2021), FiLM-Ensemble (Turkoglu et al., 2022).

Among advanced methods, BatchEnsemble seems to be one of the simplest and most flexible options. For example, FiLM-Ensemble (Turkoglu et al., 2022) requires normalization layers to be presented in the original architecture, which is not always the case for tabular MLPs. MIMO (Havasi et al., 2021), in turn, imposes additional limitations compared to BatchEnsemble. First , it requires concatenating (not stacking , as with BatchEnsemble) all k input representations, which increases the input size of the first linear layer. With the relatively high number of submodels k = 32 used in our paper, this can be an issue on datasets with a large number of features, especially when feature embeddings (Gorishniy et al., 2022) are used. For example, for k = 32 , the number of features m = 1000 , and the feature embedding size l = 32 , the input size approaches one million resulting in an extremely large first linear layer of MLP. Second , with BatchEnsemble, it is easy to explicitly materialize, analyze, and prune individual submodels. By contrast, in MIMO, all submodels are implicitly entangled within one MLP, and there is no easy way to access individual submodels.

WhyMLPs? Despite the applicability of BatchEnsemble (Wen et al., 2020) to almost any architecture, we focus specifically on MLPs. The key reason is efficiency . First, to achieve high performance, throughout the paper, we use the relatively large number of submodels k = 32 . However, the desired less-than-× k runtime overhead of BatchEnsemble typically happens only when the original model underutilizes the power of parallel computations of a given hardware. This will not be the case for attention-based models on datasets with a large number of features, as well as for retrieval-based models on datasets with a large number of objects. Second, as we show in subsection 4.3, attentionand retrieval-based models are already slow as-is. By contrast, MLPs are exceptionally efficient, to the extent that slowing them down even by an order of magnitude will still result in practical models.

Also, generally speaking, the definition of MLP suggested in subsection 3.3 and used in TabM is not special, and more advanced MLP-like backbones can be used. However, in preliminary experiments, we did not observe the benefits of more advanced backbones. Perhaps, small technical differences between backbones become less impactful in the context of parameter-efficient ensembling, at least in the scope of middle-to-large-sized datasets.

## A.2 TABM WITH FEATURE EMBEDDINGS

Notation. In this paper, we use † to mark TabM variants with the piecewise-linear embeddings (e.g. TabM † mini , TabM † , etc.).

Implementation details. In fact, there are no changes in the usage of feature embeddings compared to plain MLPs: feature embeddings are applied, and the result is flattened, before being passed to the backbones in terms of Figure 1. For example, if a dataset has m continuous features and all of them are embedded, the very first adapter R will have the shape k × md e , where d e is the feature embedding size. For TabM † mini and TabM † , we initialize the first multiplicative adapter R of the first linear layer from the standard normal distribution N (0 , 1) . The remaining details are best understood from the source code.

Efficiency. When feature embeddings are used, the simplified batching strategy from subsection 3.4 allows for more efficient implementation, when the feature embeddings are applied to the original batch size objects, and the result is simply cloned k times (compared to embedding k × batch size objects with the original batching strategy).

## A.3 HYPERPARAMETERS

We noticed that the typical optimal learning rate for TabM is higher than for MLP (note that, on each dataset, the batch size is the same for all DL models). We hypothesize that the reason is the effectively larger batch size for TabM because of how the training batches are constructed (even if the simplified batching strategy from subsection 3.4 is used).

## A.4 LIMITATIONS AND PRACTICAL CONSIDERATIONS

TabM does not introduce any new limitations compared to BatchEnsemble (Wen et al., 2020). Nevertheless, we note the following:

- The MLP backbone used in TabM is one of the simplest possible, and generally, more advanced backbones can be used. That said, some backbones may require additional care when used in TabM. For example, we did not explore backbones with normalization layers. For such layers, it is possible to allocate non-shared trainable affine transformations for each implicit submodel by adding one multiplicative and one additive adapter after the normalization layer (i.e. like in FiLM-Ensemble (Turkoglu et al., 2022)). Additional experiments are required to find the best strategy.
- For ensemble-like models, such as TabM, the notion of 'the final object embedding' changes: now, it is not a single vector, but a set of k vectors. If exactly one object embedding is required, then additional experiments may be needed to find the best way to combine k embeddings into one. The presence of multiple object embeddings can also be important for scenarios when TabM is used for solving more than one task, in particular when it is pretrained as a generic feature extractor and then reused for other tasks. The main practical guideline is that the k prediction branches should not interact with each other (e.g. through attention, pooling, etc.) and should always be trained separately.

## B EXTENDED RESULTS

This section complements section 4.

## B.1 ADDITIONAL BASELINES

In addition to the models from subsection 4.1, we consider the following baselines:

- MLP-PLR Gorishniy et al. (2022), that is, an MLP with periodic embeddings.
- ResNet (Gorishniy et al., 2021)
- SNN (Klambauer et al., 2017)
- DCNv2 (Wang et al., 2020)
- AutoInt (Song et al., 2019)
- MLP-Mixer is our adaptation of Tolstikhin et al. (2021) for tabular data.
- Trompt (Chen et al., 2023b) (our reimplementation, since there is no official implementation)

We also evaluated TabPFN (Hollmann et al., 2023), where possible. The results for this model are available only in Appendix E because this model is by design not applicable to regression tasks, which is a considerable number of our datasets. Overall, TabPFN specializes in small datasets. In line with that, the performance of TabPFN on our benchmark was not competitive.

## B.2 TASK PERFORMANCE

Figure 8 is a different version of Figure 3 with additional baselines. Overall, none of the additional baselines affect our main story.

Figure 9 is the critical difference diagram (CDD) computed over exactly the same results that were used for building Figure 3.

<!-- image -->

↓

↑

↑

Figure 8: An extended comparison of tabular models as in Figure 3. Note that the ranks (left) are computed only over the 37 datasets with random splits because ResNet, AutoInt, and MLP-Mixer were evaluated only on one 1 out of 9 datasets with domain-aware splits.

<!-- image -->

Figure 9: Critical difference diagram. The computation method is taken from the Kim et al. (2024).

## B.3 EFFICIENCY

This section complements subsection 4.3.

## Additional results.

Figure 10 complements Figure 4 by providing the training times on smaller datasets and the inference throughput on GPU with large batch sizes.

Table 3 provide the number of trainable parameters for some of the models from Figure 3.

Motivation for the benchmark setup. Comparing models under all possible kinds of budgets (task performance, the number of parameters, training time, etc.) on all possible hardware (GPU, CPU, etc.) with all possible batch sizes is rather infeasible. As such, we set a narrow goal of providing a high-level intuition on the efficiency in a transparent setting . Thus, benchmarking the transparently obtained tuned hyperparameter configurations works well for our goal. Yet, this choice also has a limitation: the hyperparameter tuning process is not aware of the efficiency budget, so it can prefer much heavier configurations even if they lead to tiny performance improvements, which will negatively affect efficiency without a good reason. Overall, we hope that the large number of datasets compensates for potentially imperfect per-dataset measurements.

## Motivation for the two setups for measuring inference throughput.

- The setup on the right side of Figure 4 simulates the online per-object predictions.
- The setup on the right side of Figure 10 simulates the offline batched computations.

<!-- image -->

↓

↑

Figure 10: ( Left ) Training time on datasets with less than 100K objects. ( Right ) Inference throughput on GPU with maximum possible batch size (i.e. the batch size depends on a model).

Table 3: Mean number of parameters with std. dev. for 7 different tuned models across all 46 datasets.

| TabM            | MLP             | FT-T            | T2G             | TabR          | ModernNCA       | SAINT               |
|-----------------|-----------------|-----------------|-----------------|---------------|-----------------|---------------------|
| 1 . 4 M 1 . 3 M | 1 . 0 M 1 . 0 M | 1 . 2 M 1 . 2 M | 2 . 1 M 1 . 6 M | 858 K 1 . 4 M | 1 . 0 M 1 . 1 M | 175 . 4 M 565 . 4 M |

±

## C DATASETS

In total, we use 46 datasets:

1. 38 datasets are taken from Gorishniy et al. (2024), which includes:
2. (a) 28 datasets from Grinsztajn et al. (2022). See the original paper for the precise dataset information.
3. (b) 10 datasets from other sources. Their properties are provided in Table 4.
2. 8 datasets from the TabReD benchmark (Rubachev et al., 2024). Their properties are provided in Table 5.

In fact, the aforementioned 38 datasets from Gorishniy et al. (2024) is only a subset of the datasets used in Gorishniy et al. (2024). Namely, we did not include the following of the remaining datasets:

- The datasets that, according to Rubachev et al. (2024), have incorrect splits and/or label leakage, including: Bike Sharing Demand , compass , electricity , SGEMM GPU kernel performance , sulfur , visualizing soil , and the weather forecasting dataset (it is replaced by the correct weather forecasting dataset from TabReD (Rubachev et al., 2024)).
- rl from (Grinsztajn et al., 2022). We observed abnormal results on these datasets. This is an anonymous dataset, which made the investigation impossible, so we removed this dataset to avoid confusion.
- yprop 4 1 from (Grinsztajn et al., 2022). Strictly speaking, this dataset was omitted due to a mistake on our side. For future work, we note that the typical performance gaps on this dataset have low absolute values in terms of RMSE. Perhaps, R 2 may be a more appropriate metric for this dataset.

±

±

±

±

±

±

Table 4: Properties of those datasets from Gorishniy et al. (2024) that are not part of Grinsztajn et al. (2022) or TabReD Rubachev et al. (2024). '# Num', '# Bin', and '# Cat' denote the number of numerical, binary, and categorical features, respectively. The table is taken from (Gorishniy et al., 2024).

| Name                | # Train   | # Validation   | # Test   |   # Num |   # Bin |   # Cat | Task type   |   Batch size |
|---------------------|-----------|----------------|----------|---------|---------|---------|-------------|--------------|
| Churn Modelling     | 6 400     | 1 600          | 2 000    |       7 |       3 |       1 | Binclass    |          128 |
| California Housing  | 13 209    | 3 303          | 4 128    |       8 |       0 |       0 | Regression  |          256 |
| House 16H           | 14 581    | 3 646          | 4 557    |      16 |       0 |       0 | Regression  |          256 |
| Adult               | 26 048    | 6 513          | 16 281   |       6 |       1 |       8 | Binclass    |          256 |
| Diamond             | 34 521    | 8 631          | 10 788   |       6 |       0 |       3 | Regression  |          512 |
| Otto Group Products | 39 601    | 9 901          | 12 376   |      93 |       0 |       0 | Multiclass  |          512 |
| Higgs Small         | 62 751    | 15 688         | 19 610   |      28 |       0 |       0 | Binclass    |          512 |
| Black Friday        | 106 764   | 26 692         | 33 365   |       4 |       1 |       4 | Regression  |          512 |
| Covertype           | 371 847   | 92 962         | 116 203  |      10 |       4 |       1 | Multiclass  |         1024 |
| Microsoft           | 723 412   | 235 259        | 241 521  |     131 |       5 |       0 | Regression  |         1024 |

Table 5: Properties of the datasets from the TabReD benchmark (Rubachev et al., 2024). '# Num', '# Bin', and '# Cat' denote the number of numerical, binary, and categorical features, respectively.

| Name               | # Train   | # Validation   | # Test   |   # Num |   # Bin |   # Cat | Task type   |   Batch size |
|--------------------|-----------|----------------|----------|---------|---------|---------|-------------|--------------|
| Sberbank Housing   | 18 847    | 4 827          | 4 647    |     365 |      17 |      10 | Regression  |          256 |
| Ecom Offers        | 109 341   | 24 261         | 26 455   |     113 |       6 |       0 | Binclass    |         1024 |
| Maps Routing       | 160 019   | 59 975         | 59 951   |     984 |       0 |       2 | Regression  |         1024 |
| Homesite Insurance | 224 320   | 20 138         | 16 295   |     253 |      23 |      23 | Binclass    |         1024 |
| Cooking Time       | 227 087   | 51 251         | 41 648   |     186 |       3 |       3 | Regression  |         1024 |
| Homecredit Default | 267 645   | 58 018         | 56 001   |     612 |       2 |      82 | Binclass    |         1024 |
| Delivery ETA       | 279 415   | 34 174         | 36 927   |     221 |       1 |       1 | Regression  |         1024 |
| Weather            | 106 764   | 42 359         | 40 840   |     100 |       3 |       0 | Regression  |         1024 |

## D IMPLEMENTATION DETAILS

## D.1 HARDWARE

Most of the experiments were conducted on a single NVIDIA A100 GPU. In rare exceptions, we used a machine with a single NVIDIA 2080 Ti GPU and Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz.

## D.2 EXPERIMENT SETUP

We mostly follow the experiment setup from Gorishniy et al. (2024). As such, some of the text below is copied from (Gorishniy et al., 2024).

Data preprocessing. For each dataset, for all DL-based solutions, the same preprocessing was used for fair comparison. For numerical features, by default, we used a slightly modified version of the quantile normalization from the Scikit-learn package (Pedregosa et al., 2011) (see the source code), with rare exceptions when it turned out to be detrimental (for such datasets, we used the standard normalization or no normalization). For categorical features, we used one-hot encoding. Binary features (i.e. the ones that take only two distinct values) are mapped to { 0 , 1 } without any further preprocessing. We completely follow Rubachev et al. (2024) on Table 5 datasets.

Training neural networks. For DL-based algorithms, we minimize cross-entropy for classification problems and mean squared error for regression problems. We use the AdamW optimizer (Loshchilov &amp;Hutter, 2019). We do not apply learning rate schedules. We do not use data augmentations. We apply global gradient clipping to 1 . 0 . For each dataset, we used a predefined dataset-specific batch size. We continue training until there are patience consecutive epochs without improvements on the validation set; we set patience = 16 for the DL models.

Hyperparameter tuning. In most cases, hyperparameter tuning is performed with the TPE sampler (typically, 50-100 iterations) from the Optuna package (Akiba et al., 2019). Hyperparameter tuning spaces for most models are provided in individual sections below (example for TabM: subsection D.9). We follow Rubachev et al. (2024) and use 25 iterations on some datasets from Table 5.

Evaluation. On a given dataset, for a given model, the tuned hyperparameters are evaluated under multiple (in most cases, 15 ) random seeds. The mean test metric and its standard deviation over these random seeds are then used to compare algorithms as described in subsection D.3.

## D.3 METRICS

We use Root Mean Squared Error for regression tasks, ROC-AUC for classification datasets from Table 5 (following Rubachev et al. (2024)), and accuracy for the rest of datasets (following Gorishniy et al. (2024)). We also tried computing ROC-AUC for all classification datasets, but did not observe any significant changes (see Figure 11), so we stuck to prior work. By default, the mean test score and its standard deviation are obtained by training a given model with tuned hyperparameters from scratch on a given dataset under 15 different random seeds.

How we compute ranks. Our method of computing ranks used in Figure 3 does not count small improvements as wins, hence the reduced range of ranks compared to other studies. Intuitively, our ranks can be considered as 'tiers'.

Recall that, on a given dataset, the performance of a given model A is expressed with the mean Amean and the standard deviation Astd of the performance score computed after the evaluation under multiple random seeds. Assuming the higher score the better, we define that the model A is better than the model B if: Amean -Astd &gt; Bmean. In other words, a model is considered better if it has a better mean score and the margin is larger than the standard deviation.

On a given dataset, when there are many models, we sort them in descending score order. Starting from the best model (with a rank equal to 1 ) we iterate over models and assign the rank 1 to all models that are no worse than the best model according to the above rule. The first model in descending order that is worse than the best model is assigned rank 2 and becomes the new reference model. We continue the process until all models are ranked. Ranks are computed independently for each dataset.

## D.4 IMPLEMENTATION DETAILS OF SUBSECTION 4.3

Applicability to large datasets. The two datasets used in Table 2 are the full versions of the 'Weather' and 'Maps Routing' datasets from the TabReD benchmark Rubachev et al. (2024). Their smaller versions with subsampled training set were already included in Table 1 and were used when building Figure 3. The validation and test sets are the same for the small and large versions of these datasets, so the task metrics are comparable between the two versions. When running models on the large versions of the datasets, we reused the hyperparameters tuned for their small versions. Thus, this experiment can be seen as a quick assessment of the applicability of several tabular DL to large datasets without a strong focus on the task performance. All models, except for FT-Transformer, were evaluated under 3 random seeds. FT-Transformer was evaluated under 1 random seed.

## D.5 IMPLEMENTATION DETAILS OF SUBSECTION 5.1

Experiment setup. This paragraph complements the description of the experiment setup in subsection 5.1. Namely, in addition to what is mentioned in the main text:

- Dropout and weight decay are turned off.
- To get representative training profiles for all models, the learning rates are tuned separately for TabM k =1 mini and TabM k =32 mini on validation sets using the usual metrics (i.e. RMSE or accuracy) as the guidance. The grid for learning rate tuning was: numpy.logspace(numpy.log10(1e-5), numpy.log10(5e-3), num=25) .

## D.6 IMPLEMENTATION DETAILS OF SUBSECTION 5.2

TabM [ G ] . Here, we clarify the implementation details for TabM [ G ] described in subsection 5.2. TabM [ G ] is obtained from a trained TabM by greedily selecting submodels from TabM starting from the best one and stopping when two conditions are simultaneously true for the first time: (1) adding

<!-- image -->

↓

↑

↑

Figure 11: Same as Figure 3, but ROC-AUC is used as the metric for all classification datasets. The two multiclass datasets presented in our benchmark are not taken into account.

any new submodel does not improve the validation metric of the collective prediction; (2) the current validation metric is already better than that of the initial model with all k submodels. To clarify, during the greedy selection, the i -th submodel is considered to be better than the j -th submodel if adding the i -th submodel to the aggregated prediction leads to better validation metrics (i.e. it is not the same as adding the submodel in the order of their individual validation metrics).

## D.7 IMPLEMENTATION DETAILS OF SUBSECTION 5.3

Figure 7 shows the mean percentage improvements (see subsection D.3) over MLP across 17 datasets: all datasets except for Covertype from Table 4, and all datasets from TabReD (Rubachev et al., 2024). We have used the dropout rate 0 . 1 and tuned the learning rate separately for each value of k . The score on each dataset is averaged over 5 seeds.

## D.8 NON-LINEAR EMBEDDINGS FOR CONTINUOUS FEATURES

Notation. We use the notation based on † and ‡ only for brevity. Any other unambiguous notation can be used in future work.

Updated piecewise-linear embeddings. We use a slightly different implementation of the piecewiselinear embeddings compared to Gorishniy et al. (2022). Architecture-wise, our implementation corresponds to the 'Q-L' and 'T-L' variations from Table 2 in Gorishniy et al. (2022) (we use the quantile-based bins for simplicity). In practice, our implementation is significantly faster and uses a different parametrization and initialization. See the source code for details.

Other models. Since it is not feasible to test all combinations of backbones and embeddings, for baselines, we stick to the embeddings used in the original papers (applies to TabR (Gorishniy et al., 2024), ExcelFormer (Chen et al., 2023a) and ModernNCA (Ye et al., 2024)). For all models with feature embeddings (including TabM, MLP, TabR, ModernNCA, ExcelFormer), the embeddingsrelated details are commented in the corresponding sections below.

## D.9 TABM

Feature embeddings. TabM † mini and TabM † are the versions of TabM with non-linear feature embeddings. TabM † mini and TabM † use the updated piecewise-linear feature embeddings mentioned in subsection D.8.

Table 6 provides the hyperparameter tuning spaces for TabM and TabMmini. Table 7 provides the hyperparameter tuning spaces for TabM † and TabM † mini .

Table 6: The hyperparameter tuning space for TabM and TabMmini. Here, (B) = { Covertype, Microsoft, Table 5 } and (A) contains all other datasets.

| Parameter           | Distribution or Value                 |
|---------------------|---------------------------------------|
| k                   | 32                                    |
| # layers            | UniformInt[1 , 5]                     |
| Width (hidden size) | UniformInt[64 , 1024]                 |
| Dropout rate        | { 0 . 0 , Uniform[0 . 0 , 0 . 5] }    |
| Learning rate       | LogUniform[1 e - 4 , 5 e - 3]         |
| Weight decay        | { 0 , LogUniform[1 e - 4 , 1 e - 1] } |
| # Tuning iterations | (A) 100 (B) 50                        |

Table 7: The hyperparameter tuning space for TabM † mini and TabM † . Here, (B) = { Covertype, Microsoft, Table 5 } and (A) contains all other datasets.

| Parameter           | Distribution or Value                 |
|---------------------|---------------------------------------|
| k                   | 32                                    |
| # layers            | UniformInt[1 , 4]                     |
| Width (hidden size) | UniformInt[64 , 1024]                 |
| Dropout rate        | { 0 . 0 , Uniform[0 . 0 , 0 . 5] }    |
| # PLE bins          | UniformInt[8 , 32]                    |
| Learning rate       | LogUniform[5 e - 5 , 3 e - 3]         |
| Weight decay        | { 0 , LogUniform[1 e - 4 , 1 e - 1] } |
| # Tuning iterations | (A) 100 (B) 50                        |

## D.10 MLP

Feature embeddings. MLP † and MLP ‡ are the versions of MLP with non-linear feature embeddings. MLP † uses the updated piecewise-linear embeddings mentioned in subsection D.8. MLP ‡ (also known as MLP-PLR) uses the periodic embeddings (Gorishniy et al., 2022). Technically, it is the PeriodicEmbeddings class from the rtdl num embeddings Python package. We tested two variations: with lite=False and lite=True . In the paper, only the former one is reported, but in the source code, the results for both are available.

Table 8, Table 9, Table 10 provide the hyperparameter tuning spaces for MLP, MLP † and MLP ‡ , respectively.

## D.11 TABR

Feature embeddings. TabR ‡ is the version of TabR with non-linear feature embeddings. TabR ‡ uses the periodic embeddings (Gorishniy et al., 2022), specifically, PeriodicEmbeddings(lite=True) from the rtdl num embeddings Python package on most datasets. On the datasets from Table 5, TabR ‡ uses the PeriodicEmbeddings(lite=True) embeddings on the Sberbank Housing and Ecom Offers datasets, and LinearReLUEmbeddings on the rest (to fit the computations into the GPU memory, following the original TabR paper).

Table 8: The hyperparameter tuning space for MLP.

| Parameter           | Distribution                          |
|---------------------|---------------------------------------|
| # layers            | UniformInt[1 , 6]                     |
| Width (hidden size) | UniformInt[64 , 1024]                 |
| Dropout rate        | { 0 . 0 , Uniform[0 . 0 , 0 . 5] }    |
| Learning rate       | LogUniform[3 e - 5 , 1 e - 3]         |
| Weight decay        | { 0 , LogUniform[1 e - 4 , 1 e - 1] } |
| # Tuning iterations | 100                                   |

Table 9: The hyperparameter tuning space for MLP † .

| Parameter                                                            | Distribution                                                                                             |
|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| # layers Width (hidden size) Dropout rate Learning rate Weight decay | UniformInt[1 , 5] UniformInt[64 , 1024] { 0 . 0 , Uniform[0 . 0 , 0 . 5] } LogUniform[3 e - 5 , 1 e - 3] |
| d embedding n bins # Tuning                                          | UniformInt[8 , 32] UniformInt[2 , 128] 100                                                               |
|                                                                      | { 0 , LogUniform[1 e - 4 , 1 e - 1] }                                                                    |
| iterations                                                           |                                                                                                          |

Table 10: The hyperparameter tuning space for MLP ‡ .

| Parameter                                                            | Distribution                                                                                             |
|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| # layers Width (hidden size) Dropout rate Learning rate Weight decay | UniformInt[1 , 5] UniformInt[64 , 1024] { 0 . 0 , Uniform[0 . 0 , 0 . 5] } LogUniform[3 e - 5 , 1 e - 3] |
| n frequencies d embedding frequency init # Tuning iterations         |                                                                                                          |
|                                                                      | { 0 , LogUniform[1 e - 4 , 1 e - 1] }                                                                    |
|                                                                      | UniformInt[16 , 96] UniformInt[16 , 32] LogUniform[1 e - 2 , 1 e 1 ] 100                                 |
| scale                                                                |                                                                                                          |

Since we follow the training and evaluation protocols from Gorishniy et al. (2024), and TabR was proposed in Gorishniy et al. (2024), we simply reuse the results for TabR. More details can be found in Appendix.D from Gorishniy et al. (2024). When tuning TabR ‡ on the datasets from Table 5, we have used 25 tuning iterations and the same tuning space as for TabR from Rubachev et al. (2024).

## D.12 FT-TRANSFORMER

We used the implementation from the ' rtdl revisiting models ' Python package. The results on datasets from Table 5 were copied from Rubachev et al. (2024), because the experiment setups are compatible.

Table 11: The hyperparameter tuning space for FT-Transformer Gorishniy et al. (2021). Here, (B) = { Covertype, Microsoft } and (A) contains all other datasets (except Table 5).

| Parameter                                            | Distribution or Value                                                                                                                                                                                           |
|------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| # blocks d token Attention FFN FFN Residual Learning | UniformInt[1 , 4] UniformInt[16 , 384] Uniform[0 . 0 , 0 . 5] Uniform[ 2 / 3 , 8 / 3 ] Uniform[0 . 0 , 0 . 5] { 0 . 0 , Uniform[0 . 0 , 0 . 2] } LogUniform[3 e - 5 , 1 e - 3] { 0 , LogUniform[1 e - 4 , 1 e - |
| dropout rate                                         |                                                                                                                                                                                                                 |
| hidden dimension expansion rate                      |                                                                                                                                                                                                                 |
| dropout rate                                         |                                                                                                                                                                                                                 |
| dropout rate                                         |                                                                                                                                                                                                                 |
| rate                                                 |                                                                                                                                                                                                                 |
| Weight decay                                         | 1] }                                                                                                                                                                                                            |
| # Tuning iterations                                  | (A) 100 (B) 50                                                                                                                                                                                                  |

## D.13 MODERNNCA

Feature embeddings. We adapted the official implementation of Ye et al. (2024). We used periodic embeddings Gorishniy et al. (2022) (specifically, PeriodicEmbeddings(lite=True) from the rtdl num embeddings Python package) for ModernNCA ‡ and no embeddings for ModernNCA. Table 12 and Table 13 provides hyperparameter tuning spaces for each ModernNCA and ModernNCA ‡ .

Table 12: The hyperparameter tuning space for ModernNCA. Here, (C) = { Table 5 } , (B) = { Covertype, Microsoft } and (A) contains all other datasets.

| Parameter                                                                | Distribution                                                                                                                                                                     |
|--------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| # blocks d block dim Dropout rate Sample rate Learning rate Weight decay | UniformInt[0 , 2] UniformInt[64 , 1024] UniformInt[64 , 1024] Uniform[0 . 0 , 0 . 5] Uniform[0 . 05 , 0 . 6] LogUniform[1 e - 5 , 1 e - 1] { 0 , LogUniform[1 e - 6 , 1 e - 3] } |

## D.14 T2G-FORMER

We adapted the implementation and hyperparameters of Yan et al. (2023) from the official repository 1 . Table 14 provides hyperparameter tuning space.

## D.15 SAINT

We completely adapted hyperparameters and protocol from Gorishniy et al. (2024) to evaluate SAINT on Grinsztajn et al. (2022) benchmark. Results on datasets from Table 4 were directly taken from Gorishniy et al. (2024). Additional details can be found in Appendix.D from Gorishniy et al. (2024). We have used a default configuration on big datasets due to the very high cost of tuning (see Table 15).

1 https://github.com/jyansir/t2g-former

Table 13: The hyperparameter tuning space for ModernNCA ‡ . Here, (C) = { Table 5 } , (B) = { Covertype, Microsoft } and (A) contains all other datasets.

| Parameter                                                                                          | Distribution                                                                                                                                                                                                             |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| # blocks d block dim Dropout rate Sample rate Learning rate Weight decay n frequencies d embedding | UniformInt[0 , 2] UniformInt[64 , 1024] UniformInt[64 , 1024] Uniform[0 . 0 , 0 . 5] Uniform[0 . 05 , 0 . 6] LogUniform[1 e - 5 , 1 e - 1] { 0 , LogUniform[1 e - 6 , 1 e - 3] } UniformInt[16 , 96] UniformInt[16 , 32] |
| # Tuning iterations                                                                                | (A) 100 (B, C) 50                                                                                                                                                                                                        |
| frequency init scale                                                                               | LogUniform[0 . 01 , 10]                                                                                                                                                                                                  |

Table 14: The hyperparameter tuning space for T2G-Former Yan et al. (2023). Here, (C) = { Table 5 } , (B) = { Covertype, Microsoft } and (A) contains all other datasets. Also, we used 50 tuning iterations on some datasets from Grinsztajn et al. (2022).

| Parameter                                                                                                                                                   | Distribution or Value                                                                                                                                                                                                                                                                            |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| # blocks d token Attention dropout rate FFN hidden dimension expansion FFN dropout rate Residual dropout rate Learning rate Col. Learning rate Weight decay | (A) UniformInt[3 , 4] (B, C) UniformInt[1 , 3] UniformInt[64 , 512] Uniform[0 . 0 , 0 . 5] (A, B) Uniform[ 2 / 3 , 8 / 3 ] (C) 4 / 3 Uniform[0 . 0 , 0 . 5] { 0 . 0 , Uniform[0 . 0 , 0 . 2] } LogUniform[3 e - 5 , 1 e - 3] LogUniform[5 e - 3 , 5 e - 2] { 0 , LogUniform[1 e - 6 , 1 e - 1] } |
| # Tuning iterations                                                                                                                                         | (A) 100 (B) 50 (C) 25                                                                                                                                                                                                                                                                            |
| rate                                                                                                                                                        |                                                                                                                                                                                                                                                                                                  |

## D.16 EXCELFORMER

Feature embeddings. ExcelFormer (Chen et al., 2023a) uses custom non-linear feature embeddings based on a GLU-style activation, see the original paper for details.

We adapted the implementation and hyperparameters of Chen et al. (2023a) from the official repository 2 . For a fair comparison with other models, we did not use the augmentation techniques from the paper in our experiments. See Table 16.

## D.17 CATBOOST, XGBOOST AND LIGHTGBM

Since our setup is directly taken from Gorishniy et al. (2024), we simply reused their results for GBDTs from the official repository 3 . Importantly, in a series of preliminary experiments, we confirmed that those results are reproducible in our instance of their setup. The details can be found in Appendix.D from Gorishniy et al. (2024). Results on datasets from Table 5 were copied from the paper (Rubachev et al., 2024).

2 https://github.com/WhatAShot/ExcelFormer

3 https://github.com/yandex-research/tabular-dl-tabr

Table 15: The default hyperparameters for SAINT (Somepalli et al., 2021) on datasets from Rubachev et al. (2024).

| Parameter                           | Value   |
|-------------------------------------|---------|
| depth                               | 2       |
| d token                             | 32      |
| n heads                             | 4       |
| d head                              | 8       |
| Attention dropout rate              | 0 . 1   |
| FFN hidden dimension expansion rate | 1       |
| FFN dropout rate                    | 0 . 8   |
| Learning rate                       | 1 e - 4 |
| Weight decay                        | 1 e - 2 |

Table 16: The hyperparameter tuning space for Excelformer Chen et al. (2023a). Here, (D) = { Homecredit, Maps Routing } , (C) = { Table 5 w/o (D) } , (B) = { Covertype, Microsoft } and (A) contains all other datasets.

| Parameter                                                                                  | Distribution or Value                                                                                                                                                                                                                                                |
|--------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| # blocks d token n heads Attention dropout FFN dropout rate Residual dropout Learning rate | (A, B) UniformInt[2 , 5] (C) UniformInt[2 , 4] (D) UniformInt[1 , 3] (A, B) { 32 , 64 , 128 , 256 } (C) { 16 , 32 , 64 } (D) { 4 , 8 , 16 , 32 } (A,B) { 4 , 8 , 16 , 32 } (C) { 4 , 8 , 16 } (D) 4 0 . 3 0 . 0 Uniform[0 . 0 , 0 . 5] LogUniform[3 e - 5 , 1 e - 3] |
| # Tuning iterations                                                                        | (A) 100 (B) 50 (C, D) 25                                                                                                                                                                                                                                             |
| rate                                                                                       |                                                                                                                                                                                                                                                                      |
| rate                                                                                       |                                                                                                                                                                                                                                                                      |
| Weight decay                                                                               | { 0 , LogUniform[1 e - 4 , 1 e - 1] }                                                                                                                                                                                                                                |

## D.18 AUTOINT

We used an implementation from Gorishniy et al. (2021) which is an adapted official implementation 4 .

## D.18.1 TABPFN

Since TabPFN accepts only less than 10K training samples we use different subsamples of size 10K for different random seeds. Also, TabPFN is not applicable to regressions and datasets with more than 100 features.

4 https://github.com/shichence/AutoInt

Table 17: The hyperparameter tuning space for AutoInt (Song et al., 2019). Here, (B) = { Covertype, Microsoft } and (A) contains all other datasets.

| Parameter                                                                                    | Distribution                                                                                                                         |
|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| # blocks d token n heads Attention dropout rate Embedding dropout Learning rate Weight decay | UniformInt[1 , 6] UniformInt[8 , 64] 2 { 0 , Uniform[0 . 0 , 0 . 5] } { 0 , Uniform[0 . 0 , 0 . 5] } LogUniform[3 e - 5 , 1 e - 3] { |
| # Tuning iterations                                                                          | (A) 100 (B) 50                                                                                                                       |
| rate                                                                                         |                                                                                                                                      |
|                                                                                              | 0 , LogUniform[1 e - 4 , 1 e - 1] }                                                                                                  |

## E PER-DATASET RESULTS WITH STANDARD DEVIATIONS

Table 18: Extended results for the main benchmark. Results are grouped by datasets. One ensemble consists of five models trained independently under different random seeds.

| churn                                                                                                                   | churn                                                                                                                                                                                                                                             | churn                                                                                                                                                                                                                                                                                                                                                  | california ↓                                                                                     | california ↓                                                                                                                                                                                                                                                          | california ↓                                                                                                                                                                                                                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Method                                                                                                                  |                                                                                                                                                                                                                                                   | Ensemble                                                                                                                                                                                                                                                                                                                                               | Method                                                                                           | Single model                                                                                                                                                                                                                                                          | Ensemble                                                                                                                                                                                                                                                                                                                                                            |
| MLP TabPFN ResNet DCN2 SNN Trompt AutoInt MLP - Mixer Excel ∗ SAINT FT - T T2G MLP ‡- lite MLP ‡ MLP † XGBoost LightGBM | ↑ Single model 0 . 8553 ± 0 . 0029 - 0 . 8545 ± 0 . 0044 0 . 8567 ± 0 . 0020 0 . 8506 ± 0 . 0051 0 . 8600 ± nan 0 . 8607 ± 0 . 0047 0 . 8592 ± 0 . 0036 0 . 8618 ± 0 . 0023 0 . 8603 ± 0 . 0029 0 . 8593 ± 0 . 0028 0 . 8613 0 . 0015             | 0 . 8582 ± 0 . 0008 0 . 8624 ± 0 . 0008 0 . 8565 ± 0 . 0035                                                                                                                                                                                                                                                                                            | MLP TabPFN ResNet DCN2 SNN Trompt AutoInt MLP - Mixer Excel ∗ SAINT                              | 0 . 4948 ± 0 . 0058 - 0 . 4915 ± 0 . 0031 0 . 4971 ± 0 . 0122 0 . 5033 ± 0 . 0075 0 . 4579 ± nan 0 . 4682 ± 0 . 0063 0 . 4746 ± 0 . 0056 0 . 4544 ± 0 . 0048 0 . 4680 ± 0 . 0048 0 . 4635 0 . 0048                                                                    | 0 . 4880 ± 0 . 0022 - 0 . 4862 ± 0 . 0017                                                                                                                                                                                                                                                                                                                           |
| CatBoost TabR TabR ‡ MNCA MNCA ‡ TabM ♠ TabM                                                                            | ± 0 . 8624 ± 0 . 0010 0 . 8624 ± 0 . 0026 0 . 8580 ± 0 . 0028 0 . 8605 ± 0 . 0022 0 . 8600 ± 0 . 0008 0 . 8582 ± 0 . 0017 0 . 8599 ± 0 . 0025 0 . 8625 ± 0 . 0021 0 . 8595 ± 0 . 0028 0 . 8606 ± 0 . 0032 0 . 8613 ± 0 . 0025 0 . 8605 ± 0 . 0016 | 0 . 8570 ± 0 . 0017 0 . 8533 ± 0 . 0033 - 0 . 8622 ± 0 . 0003 0 . 8630 ± 0 . 0005 0 . 8625 ± nan - 0 . 8598 ± 0 . 0025 - 0 . 8638 ± 0 . 0012 0 . 8640 ± 0 . 0010 0 . 8605 ± 0 . 0018 0 . 8608 ± 0 . 0013 0 . 8600 ± 0 . 0000 0 . 8588 ± 0 . 0008 0 . 8620 ± 0 . 0023 - 0 . 8615 ± 0 . 0013 0 . 8607 ± 0 . 0008 0 . 8615 ± 0 . 0005 0 . 8612 ± 0 . 0008 | FT - T T2G MLP ‡- lite MLP ‡ MLP † XGBoost LightGBM CatBoost TabR TabR ‡ MNCA MNCA ‡ TabM ♠ TabM | ± 0 . 4640 ± 0 . 0100 0 . 4652 ± 0 . 0045 0 . 4597 ± 0 . 0058 0 . 4530 ± 0 . 0029 0 . 4327 ± 0 . 0016 0 . 4352 ± 0 . 0019 0 . 4294 ± 0 . 0012 0 . 4030 ± 0 . 0023 0 . 3998 ± 0 . 0033 0 . 4239 ± 0 . 0012 0 . 4142 ± 0 . 0031 0 . 4509 ± 0 . 0032 0 . 4414 ± 0 . 0012 | 0 . 4779 ± 0 . 0022 0 . 4933 ± 0 . 0035 - 0 . 4490 ± 0 . 0028 0 . 4509 ± 0 . 0029 0 . 4350 ± nan - 0 . 4515 ± 0 . 0016 0 . 4462 ± nan 0 . 4549 ± 0 . 0006 0 . 4482 ± 0 . 0026 0 . 4491 ± 0 . 0010 0 . 4316 ± 0 . 0007 0 . 4339 ± 0 . 0008 0 . 4265 ± 0 . 0003 0 . 3964 ± 0 . 0013 - 0 . 4231 ± 0 . 0005 0 . 4071 ± 0 . 0029 0 . 4490 ± 0 . 0018 0 . 4402 ± 0 . 0001 |

±

±

±

±

| house ↓                                                                                                                                                              | house ↓                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | house ↓                                                                                                                                                                                                                                                                                                                                                                                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Method                                                                                                                                                               | Single model                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Ensemble                                                                                                                                                                                                                                                                                                                                                                                                      |
| MLP TabPFN ResNet DCN2 SNN Trompt AutoInt MLP - Mixer Excel ∗ SAINT FT - T T2G MLP ‡- lite MLP ‡ MLP † XGBoost LightGBM CatBoost TabR TabR ‡ MNCA MNCA ‡ TabM ♠ TabM | 3 . 1117 ± 0 . 0294 - 3 . 1143 ± 0 . 0258 3 . 3327 ± 0 . 0878 3 . 2176 ± 0 . 0376 3 . 0638 ± nan 3 . 2157 ± 0 . 0436 3 . 1871 ± 0 . 0519 3 . 2460 ± 0 . 0685 3 . 2424 ± 0 . 0595 3 . 1823 ± 0 . 0460 3 . 1613 ± 0 . 0320 3 . 0633 ± 0 . 0248 3 . 0775 ± 0 . 0336 3 . 0999 ± 0 . 0351 3 . 1773 ± 0 . 0102 3 . 1774 ± 0 . 0087 3 . 1172 ± 0 . 0125 3 . 0667 ± 0 . 0403 3 . 1048 ± 0 . 0410 3 . 0884 ± 0 . 0286 3 . 0704 ± 0 . 0388 3 . 0002 ± 0 . 0182 3 . 0038 ± 0 . 0097 | 3 . 0706 ± 0 . 0140 - 3 . 0706 ± 0 . 0098 3 . 1303 ± 0 . 0410 3 . 1320 ± 0 . 0155 - 3 . 1261 ± 0 . 0095 3 . 0184 ± 0 . 0086 3 . 1097 ± nan - 3 . 0974 ± 0 . 0334 3 . 0982 ± nan 3 . 0170 ± 0 . 0070 3 . 0268 ± 0 . 0170 3 . 0401 ± 0 . 0071 3 . 1644 ± 0 . 0068 3 . 1672 ± 0 . 0050 3 . 1058 ± 0 . 0022 2 . 9958 ± 0 . 0270 - 3 . 0538 ± 0 . 0072 3 . 0149 ± 0 . 0308 2 . 9796 ± 0 . 0024 2 . 9906 ± 0 . 0026 |

±

## diamond ↓

Single model

0

0012

.

1404

0

.

-

±

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

0

.

0

.

.

0

.

0

0

.

0

.

.

0

.

0

.

0

.

0

0

.

.

0

.

0

.

0

.

0

.

0

.

0

.

1396

1473

1420

1391

1400

1392

1766

1376

1369

1372

1342

1337

1323

1359

1368

1335

1327

1333

1370

1327

1342

1310

1309

1323

1315

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

.

0

.

0

.

0032

0029

0057

nan

0

0025

.

.

0

0

.

0

.

0

0

.

.

0

.

0

.

0

.

0

.

0

0

.

0

.

.

0

0

0014

0023

0013

0019

0011

0008

0010

0010

0002

0004

0006

0010

.

.

0

0

0

0

0

0

0018

0013

.

0012

.

0017

.

0007

.

0008

.

0007

.

0006

Method

MLP

ResNet

TabPFN

DCN2

Trompt

SNN

AutoInt

-

Mixer

MLP

Excel

∗

SAINT

-

T2G

T

FT

‡-

MLP

‡

MLP

†

XGBoost

MLP

LightGBM

TabR

CatBoost

TabR

‡

MNCA

‡

MNCA

TabM

♠

TabM

TabM[G]

mini

TabM

TabM

†

mini

lite

±

Ensemble

0

.

1362

-

±

0

.

0

.

.

0

-

0

.

.

0

0

.

-

0

.

.

0

0

.

0

.

0

.

0

.

0

0

.

0

.

.

-

.

0

0

.

0

.

0

.

-

0

.

0

.

1361

1424

1374

1361

1378

1712

1360

1346

1325

1317

1301

1358

1363

1327

1311

1348

1315

1327

1307

1317

1312

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

.

0

.

0

.

0003

0011

0008

0

.

0020

0

0

.

0008

.

0004

nan

0

nan

.

0002

0

.

0004

0

0

0

0

0

0

0

0

0

0

0

0

.

0003

.

0005

.

.

0001

.

0001

.

0005

0004

.

0005

.

.

.

.

.

0006

0004

0002

0002

0001

Method

MLP

ResNet

TabPFN

DCN2

Trompt

SNN

AutoInt

-

∗

MLP

Mixer

Excel

SAINT

-

T2G

FT

T

‡-

MLP

‡

MLP

†

XGBoost

MLP

LightGBM

TabR

CatBoost

TabR

‡

MNCA

‡

MNCA

TabM

♠

TabM

TabM[G]

mini

TabM

TabM

†

mini

Method

MLP

ResNet

TabPFN

DCN2

Trompt

SNN

AutoInt

-

∗

MLP

Mixer

Excel

SAINT

-

T2G

FT

T

‡-

MLP

‡

MLP

†

XGBoost

MLP

LightGBM

TabR

CatBoost

TabR

‡

MNCA

‡

MNCA

TabM

♠

TabM

TabM[G]

mini

TabM

TabM

†

mini

lite

lite Single model

adult ↑

0

0018

.

8540

0

.

-

±

0

.

0

0

.

0

.

0

.

0

.

.

0

.

0

0

.

0

0

0

0

.

.

.

.

0

.

.

0

.

0

.

0

.

0

0

.

.

0

.

0

0

.

.

0

.

0

.

0

0

.

0011

.

0

0

.

0

0011

0009

nan

0

0013

.

.

0

0016

.

0

0

.

0

0

0

0

0

0

0

0

0

0

0

0

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

0019

0024

.

0011

.

0015

.

0007

.

0011

.

.

.

.

.

.

.

.

.

.

0009

0007

0006

0012

0022

0011

0018

0008

0011

0008

±

0

.

0011

±

0

.

0010

±

0

.

0007

.

8554

8582

8582

8590

8598

8592

8613

8588

8601

8601

8693

8694

8603

8713

8720

8714

8646

8699

8677

8717

8582

8575

8572

8598

8700

otto ↑

Single model

0

0022

.

8175

0

.

-

±

0

.

0

0

.

0

.

.

.

0

.

0

0

.

0

0

0

0

0

0

.

.

.

.

.

0

.

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

8174

8087

8064

8093

8092

8050

8102

8133

8119

8161

8190

8189

8205

8302

8297

8250

8179

8246

8275

8265

8268

8275

8254

8282

8342

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

.

0

.

0

.

0021

0021

0020

nan

0

.

0040

0

.

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0034

.

0022

.

.

0018

.

0019

0033

.

0021

.

0015

.

.

.

.

.

.

.

.

.

.

.

.

.

0021

0009

0011

0013

0022

0018

0012

0015

0014

0014

0022

0014

0012

Ensemble

0

.

8559

-

±

0

.

0

.

.

0

-

0

.

.

0

0

.

-

0

.

.

0

0

.

0

.

0

0

.

0

.

0

.

0

.

.

-

0

0

.

.

0

0

.

.

-

0

.

8604

0

±

.

8701

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

8562

8603

8593

8612

8617

8641

8608

8622

8702

8704

8616

8721

8723

8723

8680

8696

8742

8588

8583

±

Ensemble

0

±

.

8222

0

.

±

0

.

7408

0

.

0

.

-

0

.

.

0

0

.

-

0

.

.

0

0

.

0

.

0

0

.

0

.

0

.

0

.

.

-

0

0

.

.

0

0

.

.

-

0

.

0

8299

.

8356

8208

8198

8156

8111

8136

8220

8221

8272

8271

8253

8290

8316

8316

8268

8236

8313

8304

8300

8284

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

0

0

.

0011

.

0006

.

0012

0

.

0002

0

0

.

0002

.

0004

nan

0

nan

.

0011

0

.

0006

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

.

.

.

.

.

.

.

.

.

.

.

.

0008

0006

0004

0002

0007

0019

0003

0006

0003

0004

0000

0003

.

0007

.

0006

.

0028

.

.

0013

0023

0

0

.

0010

.

0020

nan

0

nan

.

0013

0

.

0015

0

0

0

0

0

0

0

0

0

0

0

0

.

.

.

.

.

.

.

.

.

.

.

.

0000

0006

0013

0008

0002

0009

0006

0006

0007

0005

0005

0004

↑

| Method            | higgs-small Single model                | Ensemble                              | Method              | black-friday Single model               | Ensemble                              |
|-------------------|-----------------------------------------|---------------------------------------|---------------------|-----------------------------------------|---------------------------------------|
| MLP               | 0 . 7180 0 . 0027                       | 0 . 7192 0 . 0005                     | MLP                 | 0 . 6955 0 . 0004                       | 0 . 6942 0 . 0002                     |
| TabPFN            | ± -                                     | ± 0 . 6727 ± 0 . 0034                 | TabPFN              | ± -                                     | ± -                                   |
| ResNet            | 0 . 7256 ± 0 . 0020                     | 0 . 7307 0 . 0001                     | ResNet              | 0 . 6929 0 . 0008                       | 0 . 6907 ± 0 . 0002                   |
| DCN2              | 0 . 7164 ± 0 . 0030                     | ± 0 . 7237 ± 0 . 0011                 | DCN2                | ± 0 . 6968 ± 0 . 0013                   | 0 . 6936 ± 0 . 0007                   |
| SNN               | 0 . 7142 ± 0 . 0024                     | 0 . 7171 ± 0 . 0020                   | SNN                 | 0 . 6996 ± 0 . 0013                     | 0 . 6978 ± 0 . 0004                   |
| Trompt            | 0 . 7262 ± nan                          | -                                     | Trompt              | 0 . 6983 ± nan                          | -                                     |
| AutoInt           | 0 . 7240 ± 0 . 0028                     | 0 . 7287 ± 0 . 0008                   | AutoInt             | 0 . 6994 ± 0 . 0082                     | 0 . 6927 ± 0 . 0021                   |
| MLP - Mixer       | 0 . 7248 ± 0 . 0023                     | 0 . 7334 ± 0 . 0007                   | MLP - Mixer         | 0 . 6905 ± 0 . 0021                     | 0 . 6851 ± 0 . 0011                   |
| Excel ∗           | 0 . 7262 ± 0 . 0017                     | 0 . 7329 ± nan                        | Excel ∗             | 0 . 6947 ± 0 . 0016                     | 0 . 6908 ± nan                        |
| SAINT             | 0 . 7236 ± 0 . 0019                     | -                                     | SAINT               | 0 . 6934 ± 0 . 0009                     | -                                     |
| FT - T            | 0 . 7281 ± 0 . 0016                     | 0 . 7334 ± 0 . 0013                   | FT - T              | 0 . 6987 ± 0 . 0192                     | 0 . 6879 ± 0 . 0023                   |
| T2G               | 0 . 7352 ± 0 . 0037                     | 0 . 7400 ± nan                        | T2G                 | 0 . 6887 ± 0 . 0046                     | 0 . 6832 ± nan                        |
| MLP ‡- lite       | 0 . 7260 ± 0 . 0017                     | 0 . 7304 ± 0 . 0008                   | MLP ‡- lite         | 0 . 6849 ± 0 . 0006                     | 0 . 6824 ± 0 . 0002                   |
| MLP ‡             | 0 . 7261 ± 0 . 0010                     | 0 . 7270 ± 0 . 0003                   | MLP ‡               | 0 . 6857 ± 0 . 0004                     | 0 . 6838 ± 0 . 0002                   |
| MLP †             | 0 . 7210 0 . 0016                       | 0 . 7252 0 . 0005                     | MLP †               | 0 . 6836 0 . 0006                       | 0 . 6812 0 . 0002                     |
| XGBoost           | ± 0 . 7246 0 . 0015                     | ± 0 . 7264 0 . 0013                   | XGBoost             | ± 0 . 6806 ± 0 . 0001                   | ± 0 . 6805 ± 0 . 0000                 |
| LightGBM          | ± 0 . 7256 ± 0 . 0009                   | ± 0 . 7263 ± 0 . 0007                 | LightGBM            | 0 . 6799 ± 0 . 0003                     | 0 . 6795 ± 0 . 0001                   |
| CatBoost          | 0 . 7260 ± 0 . 0011                     | 0 . 7273 ± 0 . 0010                   | CatBoost            | 0 . 6822 ± 0 . 0003                     | 0 . 6813 ± 0 . 0002                   |
| TabR              | 0 . 7223 ± 0 . 0010                     | 0 . 7257 ± 0 . 0008                   | TabR                | 0 . 6899 ± 0 . 0004                     | 0 . 6883 ± 0 . 0002                   |
| TabR ‡            | 0 . 7294 ± 0 . 0014                     | -                                     | TabR ‡              | 0 . 6761 ± 0 . 0009                     | -                                     |
| MNCA              | 0 . 7263 ± 0 . 0023                     | 0 . 7292 ± 0 . 0006                   | MNCA                | 0 . 6893 ± 0 . 0004                     | 0 . 6883 ± 0 . 0000                   |
| MNCA ‡            | 0 . 7300 ± 0 . 0020                     | 0 . 7348 ± 0 . 0008                   | MNCA ‡              | 0 . 6885 ± 0 . 0007                     | 0 . 6863 ± 0 . 0003                   |
| TabM ♠            | 0 . 7383 ± 0 . 0028                     | 0 . 7409 0 . 0010                     | TabM ♠              | 0 . 6875 ± 0 . 0015                     | 0 . 6866 ± 0 . 0003                   |
| TabM              | 0 . 7394 ± 0 . 0018                     | ± 0 . 7409 ± 0 . 0008                 | TabM                | 0 . 6869 ± 0 . 0004                     | 0 . 6865 ± 0 . 0001                   |
| TabM[G]           | 0 . 7392 ± 0 . 0016                     | -                                     | TabM[G]             | 0 . 6865 ± 0 . 0005                     | -                                     |
| TabM              | 0 . 7338 ± 0 . 0011                     | 0 . 7345 ± 0 . 0008                   | TabM                |                                         | 0 . 6856 0 . 0003                     |
| mini TabM † mini  | 0 . 7361 ± 0 . 0011                     | 0 . 7383 ± 0 . 0008                   | mini TabM † mini    | 0 . 6863 ± 0 . 0006 0 . 6781 ± 0 . 0004 | ± 0 . 6773 ± 0 . 0001                 |
| covtype2 ↑        | covtype2 ↑                              | covtype2 ↑                            | microsoft ↓         | microsoft ↓                             | microsoft ↓                           |
| Method            | Single model                            | Ensemble                              | Method              | Single model                            | Ensemble                              |
| MLP               | 0 . 9630 0 . 0012                       | 0 . 9664 0 . 0004                     | MLP                 | 0 . 7475 ± 0 . 0003                     | 0 . 7460 ± 0 . 0003                   |
| TabPFN            | ± -                                     | ± 0 . 7606 ± 0 . 0022                 | TabPFN              | -                                       | -                                     |
| ResNet            | 0 . 9638 ± 0 . 0005                     | 0 . 9685 ± 0 . 0003                   | ResNet              | 0 . 7472 ± 0 . 0004 0 . 7499 0 . 0003   | 0 . 7452 ± 0 . 0004 0 . 7477 0 . 0001 |
| DCN2 SNN          | 0 . 9622 ± 0 . 0019 0 . 9636 0 . 0010   | 0 . 9673 ± 0 . 0011 0 . 9677 0 . 0002 | DCN2 SNN            | ± 0 . 7488 ± 0 . 0004                   | ± 0 . 7470 0 . 0001                   |
| Trompt            | ± 0 . 9286 nan                          | ±                                     |                     |                                         | ± -                                   |
|                   | ±                                       | -                                     | Trompt              | 0 . 7476 ± nan                          | 0 . 7455 0 . 0002                     |
| AutoInt           | 0 . 9614 ± 0 . 0016                     | 0 . 9696 ± 0 . 0005                   | AutoInt             | 0 . 7482 ± 0 . 0005                     | ± 0 . 7436 0 . 0001                   |
| MLP - Mixer ∗     | 0 . 9663 ± 0 . 0019 0 . 9606 0 . 0018   | 0 . 9699 ± 0 . 0014                   | MLP - Mixer Excel ∗ | 0 . 7482 ± 0 . 0008 0 . 7479 0 . 0007   | ± 0 . 7442 nan                        |
| Excel             | ± 0 . 9669 0 . 0010                     | 0 . 9670 ± nan -                      | SAINT               | ± 0 . 7625 0 . 0066                     | ± -                                   |
| SAINT FT - T      | ± 0 . 9698 0 . 0008                     | 0 . 9731 ± 0 . 0006                   | FT - T              | ± 0 . 7460 0 . 0007                     | 0 . 7422 ± 0 . 0004                   |
| T2G               | ± 0 . 9668 ± 0 . 0008                   | 0 . 9708 ± nan                        | T2G                 | ± 0 . 7460 ± 0 . 0006                   | 0 . 7427 ± nan                        |
| MLP ‡- lite       | 0 . 9690 ± 0 . 0008                     | 0 . 9721 ± 0 . 0006                   | MLP ‡- lite         | 0 . 7446 ± 0 . 0002                     | 0 . 7434 ± 0 . 0002                   |
| MLP ‡             | 0 . 9713 0 . 0006                       | 0 . 9758 ± 0 . 0000                   | MLP ‡               | 0 . 7444 ± 0 . 0003                     | 0 . 7429 ± 0 . 0001                   |
| †                 | ±                                       |                                       | †                   |                                         | 0 . 7448 0 . 0001                     |
| MLP               | 0 . 9697 ± 0 . 0008                     | 0 . 9721 ± 0 . 0005                   | MLP                 | 0 . 7465 ± 0 . 0005                     | ±                                     |
| XGBoost           | 0 . 9710 ± 0 . 0002                     | 0 . 9713 ± 0 . 0000                   | XGBoost             | 0 . 7413 ± 0 . 0001                     | 0 . 7410 ± 0 . 0000                   |
| LightGBM          | 0 . 9709 ± 0 . 0003                     | -                                     | LightGBM            | 0 . 7417 ± 0 . 0001                     | 0 . 7413 ± 0 . 0000                   |
| CatBoost          | 0 . 9670 ± 0 . 0003                     | 0 . 9680 ± 0 . 0002                   | CatBoost            | 0 . 7412 ± 0 . 0001                     | 0 . 7406 ± 0 . 0000                   |
| TabR              | 0 . 9737 ± 0 . 0005                     | 0 . 9745 ± 0 . 0006                   | TabR                | 0 . 7503 ± 0 . 0006                     | 0 . 7485 ± 0 . 0002                   |
|                   | 0 . 9752 0 . 0003                       | -                                     | TabR ‡              | 0 . 7501 ± 0 . 0005                     | -                                     |
| TabR ‡ MNCA       | ± 0 . 9724 0 . 0003                     | 0 . 9729 ± 0 . 0001                   | MNCA                | 0 . 7458 ± 0 . 0003                     | 0 . 7448 ± 0 . 0002                   |
| MNCA ‡            | ± 0 . 9747 ± 0 . 0002                   | 0 . 9747 ± 0 . 0002                   | MNCA ‡              | 0 . 7460 ± 0 . 0008                     | 0 . 7435 ± 0 . 0004                   |
| TabM ♠            | 0 . 9712 ± 0 . 0008                     | 0 . 9729 ± 0 . 0003                   | TabM ♠              | 0 . 7434 ± 0 . 0003                     | 0 . 7424 ± 0 . 0001                   |
| TabM              | 0 . 9735 ± 0 . 0004                     | 0 . 9743 ± 0 . 0001                   | TabM                | 0 . 7432 ± 0 . 0004                     | 0 . 7426 ± 0 . 0001                   |
| TabM[G] TabM mini | 0 . 9730 ± 0 . 0005 0 . 9710 ± 0 . 0007 | - 0 . 9727 ± 0 . 0002                 | TabM[G] TabM mini   | 0 . 7432 ± 0 . 0004 0 . 7436 ± 0 . 0002 | - 0 . 7430 ± 0 . 0002                 |
| TabM † mini       | 0 . 9755 ± 0 . 0003                     | 0 . 9762 ± 0 . 0001                   | TabM † mini         | 0 . 7423 ± 0 . 0002                     | 0 . 7416 ± 0 . 0001                   |

↓

Table 19: Extended results for Grinsztajn et al. (2022) benchmark. Results are grouped by datasets. One ensemble consists of five models trained independently with different random seeds.

wine ↑

±

| wine                                      | wine                                                                                                                                        | wine                                                        |
|-------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| Method                                    | Single model                                                                                                                                | Ensemble                                                    |
| MLP TabPFN ResNet DCN2                    | 0 . 7778 ± 0 . 0153 -                                                                                                                       | 0 . 7907 ± 0 . 0117 0 . 7908 ± 0 . 0063                     |
| SNN Trompt                                | 0 . 7710 ± 0 . 0 . 7492 ± 0 . 0 . 7818 ± 0 .                                                                                                | 0 . 7839 ± 0 . 0083 0 . 7764 ± 0 . 0095 0 . 7994 ± 0 . 0097 |
| AutoInt MLP - Excel ∗ SAINT FT - T T2G ‡- | 0137 0147 0143 0 . 7818 ± 0 . 0081 0 . 7745 ± 0 . 0144                                                                                      | - 0 . 7909 ± 0 . 0160                                       |
| MLP LightGBM TabR TabR ‡ MNCA MNCA        | 0 . 7769 ± 0 . 0149 0 . 7631 ± 0 . 0171 0 . 7684 ± 0 . 0144 0 . 7755 ± 0 . 0133 0 . 7733 ± 0 . 0118 0 . 7803 ± 0 . 0157 0 . 7733 ± 0 . 0185 |                                                             |
| ‡                                         | 0 . 7867 ± 0 . 0113 0 . 7961 ± 0 . 0136                                                                                                     |                                                             |
| Mixer                                     |                                                                                                                                             | 0 . 7950 ± 0 . 0087                                         |
|                                           |                                                                                                                                             | 0 . 7765 ± 0 . 0121                                         |
|                                           |                                                                                                                                             | -                                                           |
|                                           |                                                                                                                                             | 0 . 7894 ± 0 . 0083                                         |
|                                           |                                                                                                                                             | 0 . 7933 ± 0 . 0137                                         |
| MLP lite                                  |                                                                                                                                             | 0 . 7964 ± 0 . 0146                                         |
| MLP ‡                                     |                                                                                                                                             | 0 . 7856 ± 0 . 0160                                         |
| †                                         | 0 . 7814 ± 0 . 0132                                                                                                                         | 0 . 7919 ± 0 . 0098                                         |
| XGBoost                                   | 0 . 7949 ± 0 . 0178 0 . 7890 0 . 0160                                                                                                       | 0 . 8010 ± 0 . 0186                                         |
| CatBoost                                  | ± 0 . 7994 ± 0 . 0131                                                                                                                       | 0 . 7929 ± 0 . 0106 0 . 8057 ± 0 . 0098                     |
|                                           | 0 . 7936 ± 0 . 0114                                                                                                                         |                                                             |
|                                           |                                                                                                                                             | 0 . 8055 ± 0 . 0057                                         |
|                                           | 0 . 7911 ± 0 . 0135                                                                                                                         | 0 . 8005 ± 0 . 0121                                         |
|                                           | 0 . 7804 ± 0 . 0148                                                                                                                         | -                                                           |
| ♠                                         |                                                                                                                                             | 0 . 7953 ± 0 . 0114                                         |
| TabM                                      |                                                                                                                                             | 0 . 8011 ± 0 . 0084                                         |
| TabM                                      | 0 . 7943 ± 0 . 0124                                                                                                                         | 0 . 7985 ± 0 . 0139                                         |
| TabM[G]                                   | 0 . 7879 ± 0 . 0161                                                                                                                         | -                                                           |
| TabM mini                                 | 0 . 7890 ± 0 . 0130                                                                                                                         | 0 . 7937 ± 0 . 0103                                         |
| TabM † mini                               | 0 . 7839 0 . 0169                                                                                                                           | 0 . 7917 0 . 0143                                           |

±

## analcatdata supreme ↓

| Method      | Single model          | Ensemble              |
|-------------|-----------------------|-----------------------|
| MLP TabPFN  | 0 . 0782 ± 0 . 0081 - | 0 . 0766 ± 0 . 0090 - |
| ResNet      | 0 . 0852 ± 0 . 0076   | 0 . 0823 ± 0 . 0078   |
| DCN2        | 0 . 0811 ± 0 . 0137   | 0 . 0759 ± 0 . 0086   |
| SNN         | 0 . 0826 ± 0 . 0096   | 0 . 0779 ± 0 . 0098   |
| Trompt      | 0 . 0782 ± 0 . 0095   | -                     |
| AutoInt     | 0 . 0783 ± 0 . 0078   | 0 . 0768 ± 0 . 0083   |
| MLP - Mixer | 0 . 0770 ± 0 . 0082   | 0 . 0759 ± 0 . 0081   |
| Excel ∗     | 0 . 0796 ± 0 . 0101   | 0 . 0776 ± 0 . 0101   |
| SAINT       | 0 . 0773 ± 0 . 0078   | -                     |
| FT - T      | 0 . 0787 ± 0 . 0086   | 0 . 0775 ± 0 . 0091   |
| T2G         | 0 . 0775 ± 0 . 0081   | 0 . 0763 ± 0 . 0084   |
| MLP ‡- lite | 0 . 0798 ± 0 . 0088   | 0 . 0769 ± 0 . 0092   |
| MLP ‡       | 0 . 0786 ± 0 . 0073   | 0 . 0720 ± 0 . 0053   |
| MLP †       | 0 . 0774 ± 0 . 0064   | 0 . 0759 ± 0 . 0063   |
| XGBoost     | 0 . 0801 ± 0 . 0126   | 0 . 0774 ± 0 . 0107   |
| LightGBM    | 0 . 0778 ± 0 . 0115   | 0 . 0767 ± 0 . 0110   |
| CatBoost    | 0 . 0780 ± 0 . 0067   | 0 . 0734 ± 0 . 0022   |
| TabR        | 0 . 0803 ± 0 . 0066   | 0 . 0759 ± 0 . 0046   |
| TabR ‡      | 0 . 0807 ± 0 . 0088   |                       |
| MNCA        | 0 . 0809 ± 0 . 0072   | - 0 . 0784 ± 0 . 0062 |
| MNCA ‡      | 0 . 0825 ± 0 . 0090   | 0 . 0793 ± 0 . 0072   |
| TabM ♠      | 0 . 0777 ± 0 . 0099   | 0 . 0769 ± 0 . 0105   |
| TabM        | 0 . 0786 ± 0 . 0055   | 0 . 0781 ± 0 . 0054   |
| TabM[G]     | 0 . 0808 ± 0 . 0063   | -                     |
| TabM mini   | 0 . 0773 ± 0 . 0077   | 0 . 0763 ± 0 . 0077   |
| TabM † mini | 0 . 0764 ± 0 . 0071   | 0 . 0749 ± 0 . 0076   |

## phoneme ↑

±

| Method      | Single model          | Ensemble                                |
|-------------|-----------------------|-----------------------------------------|
| MLP TabPFN  | 0 . 8525 ± 0 . 0126 - | 0 . 8635 ± 0 . 0099 0 . 8684 ± 0 . 0050 |
| ResNet      | 0 . 8456 ± 0 . 0121   | 0 . 8504 ± 0 . 0066                     |
| DCN2        | 0 . 8342 ± 0 . 0151   | 0 . 8543 ± 0 . 0118                     |
| SNN         | 0 . 8596 ± 0 . 0124   | 0 . 8687 ± 0 . 0080                     |
| Trompt      | 0 . 8465 ± 0 . 0205   | -                                       |
| AutoInt     | 0 . 8623 ± 0 . 0138   | 0 . 8754 ± 0 . 0095                     |
| MLP - Mixer | 0 . 8629 ± 0 . 0123   | 0 . 8757 ± 0 . 0095                     |
| Excel ∗     | 0 . 8551 ± 0 . 0092   | 0 . 8711 ± 0 . 0081                     |
| SAINT       | 0 . 8657 ± 0 . 0130   | -                                       |
| FT - T      | 0 . 8667 ± 0 . 0127   | 0 . 8795 ± 0 . 0093                     |
| T2G         | 0 . 8672 ± 0 . 0166   | 0 . 8765 ± 0 . 0141                     |
| MLP ‡- lite | 0 . 8742 ± 0 . 0120   | 0 . 8861 ± 0 . 0071                     |
| MLP ‡       | 0 . 8757 ± 0 . 0118   | 0 . 8856 ± 0 . 0065                     |
| MLP †       | 0 . 8647 ± 0 . 0098   | 0 . 8761 ± 0 . 0076                     |
| XGBoost     | 0 . 8682 ± 0 . 0174   | 0 . 8771 ± 0 . 0156                     |
| LightGBM    | 0 . 8702 ± 0 . 0129   | 0 . 8733 ± 0 . 0126                     |
| CatBoost    | 0 . 8827 ± 0 . 0117   | 0 . 8897 ± 0 . 0055                     |
| TabR        | 0 . 8781 ± 0 . 0096   | 0 . 8840 ± 0 . 0054                     |
| TabR ‡      | 0 . 8772 ± 0 . 0087   | -                                       |
| MNCA        | 0 . 8835 ± 0 . 0079   | 0 . 8861 ± 0 . 0057                     |
| MNCA ‡      | 0 . 8828 ± 0 . 0082   | 0 . 8925 ± 0 . 0056                     |
| TabM ♠      | 0 . 8701 ± 0 . 0167   | 0 . 8766 ± 0 . 0128                     |
| TabM        | 0 . 8831 ± 0 . 0121   | 0 . 8880 ± 0 . 0108                     |
| TabM[G]     | 0 . 8762 ± 0 . 0144   | -                                       |
| TabM mini   | 0 . 8803 ± 0 . 0098   | 0 . 8842 ± 0 . 0067                     |
| TabM † mini | 0 . 8780 0 . 0119     | 0 . 8817 0 . 0101                       |

±

## Mercedes Benz Greener Manufacturing ↓

| Method                                                                                                                                                                         | Single model                                                                                                                                                                                                                                                                                                    | Ensemble                                                                                                                                                                                                                                                                    |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MLP TabPFN ResNet DCN2 SNN Trompt AutoInt MLP - Excel ∗ SAINT FT - T T2G MLP ‡- lite MLP ‡ MLP † XGBoost CatBoost TabR TabR ‡ MNCA MNCA ‡ TabM ♠ TabM TabM[G] TabM mini TabM † | 8 . 3045 ± 0 . 8708 - ± 8 . 3045 ± 0 . 8708 8 . 3045 ± 0 . 8708 8 . 2177 ± 0 . 8175 8 . 2078 ± 0 . 8231 8 . 1629 ± 0 . 8193 8 . 3506 ± 0 . 8149 8 . 3187 ± 0 . 8186 8 . 2557 ± 0 . 8602 8 . 2557 ± 0 . 8602 8 . 2215 ± 0 . 8940 8 . 2052 ± 0 . 9043 8 . 2235 ± 0 . 8867 8 . 2075 ± 0 . 9185 8 . 2075 ± 0 . 9185 | 8 . 2682 ± 0 . 8992 - ± 8 . 2682 ± 0 . 8992 8 . 2682 ± 0 . 8992 8 . 2092 ± 0 . 8458 8 . 1618 ± 0 . 8566 8 . 1554 ± 0 . 8439 8 . 2694 ± 0 . 8399 - 8 . 1771 ± 0 . 8710 8 . 1771 ± 0 . 8710 8 . 1995 ± 0 . 9130 8 . 1965 ± 0 . 9306 - 8 . 1986 ± 0 . 9442 8 . 1986 ± 0 . 9442 |
| Mixer LightGBM                                                                                                                                                                 | 8 . 4434 ± 0 . 7982 8 . 3540 ± 0 . 8314 8 . 2718 ± 0 . 8152 8 . 3409 ± 0 . 9840 8 . 4001 ± 0 . 9256 8 . 2860 ± 0 . 8656 8 . 2244 ± 0 . 8514 8 . 3556 ± 0 . 9566 8 . 2252 ± 0 . 8617 8 . 2120 ± 0 . 8485 8 . 3045 0 . 8708                                                                                       | 8 . 3178 ± 0 . 8482 8 . 3021 ± 0 . 8579 8 . 2236 ± 0 . 8479 - 8 . 3237 ± 0 . 9658 8 . 2398 ± 0 . 9023 8 . 1918 ± 0 . 9387 - 8 . 1616 ± 0 . 8834 8 . 1654 ± 0 . 9339 8 . 2682 0 . 8992                                                                                       |

KDDCup09 upselling ↑

±

| Method      | Single model          | Ensemble              |
|-------------|-----------------------|-----------------------|
| MLP TabPFN  | 0 . 7759 ± 0 . 0137 - | 0 . 7806 ± 0 . 0125 - |
| ResNet      | 0 . 7811 ± 0 . 0124   | 0 . 7861 ± 0 . 0109   |
| DCN2        | 0 . 7850 ± 0 . 0161   | 0 . 7884 ± 0 . 0135   |
| SNN         | 0 . 7884 ± 0 . 0122   | 0 . 7940 ± 0 . 0116   |
| Trompt      | 0 . 7994 ± 0 . 0055   | -                     |
| AutoInt     | 0 . 8004 ± 0 . 0075   | 0 . 8037 ± 0 . 0063   |
| MLP - Mixer | 0 . 7979 ± 0 . 0105   | 0 . 8010 ± 0 . 0094   |
| Excel ∗     | 0 . 7903 ± 0 . 0074   | 0 . 7939 ± 0 . 0099   |
| SAINT       | 0 . 7942 ± 0 . 0112   | -                     |
| FT - T      | 0 . 7957 ± 0 . 0127   | 0 . 7960 ± 0 . 0139   |
| T2G         | 0 . 8037 ± 0 . 0100   | 0 . 7988 ± 0 . 0084   |
| MLP ‡- lite | 0 . 7962 ± 0 . 0093   | 0 . 7995 ± 0 . 0105   |
| MLP ‡       | 0 . 8005 ± 0 . 0097   | 0 . 8032 ± 0 . 0117   |
| MLP †       | 0 . 7925 ± 0 . 0123   | 0 . 7963 ± 0 . 0089   |
| XGBoost     | 0 . 7930 ± 0 . 0108   | 0 . 7950 ± 0 . 0102   |
| LightGBM    | 0 . 7932 ± 0 . 0119   | 0 . 7969 ± 0 . 0115   |
| CatBoost    | 0 . 7992 ± 0 . 0117   | 0 . 8010 ± 0 . 0121   |
| TabR        | 0 . 7838 ± 0 . 0136   | 0 . 7859 ± 0 . 0167   |
| TabR ‡      | 0 . 7908 ± 0 . 0123   | -                     |
| MNCA        | 0 . 7939 ± 0 . 0097   | 0 . 7989 ± 0 . 0115   |
| MNCA ‡      | 0 . 7960 ± 0 . 0131   | 0 . 8008 ± 0 . 0110   |
| TabM ♠      | 0 . 8002 ± 0 . 0103   | 0 . 8021 ± 0 . 0074   |
| TabM        | 0 . 8024 ± 0 . 0111   | 0 . 8054 ± 0 . 0123   |
| TabM[G]     | 0 . 7988 ± 0 . 0118   | -                     |
| TabM mini   | 0 . 7971 ± 0 . 0117   | 0 . 7982 ± 0 . 0107   |
| TabM † mini | 0 . 8024 0 . 0075     | 0 . 8035 0 . 0088     |

±

wine quality ↓

| Method      | Single model          | Ensemble              |
|-------------|-----------------------|-----------------------|
| MLP TabPFN  | 0 . 6707 ± 0 . 0178 - | 0 . 6530 ± 0 . 0152 - |
| ResNet      | 0 . 6687 ± 0 . 0166   | 0 . 6543 ± 0 . 0170   |
| DCN2        | 0 . 7010 ± 0 . 0171   | 0 . 6699 ± 0 . 0139   |
| SNN         | 0 . 6604 ± 0 . 0174   | 0 . 6245 ± 0 . 0140   |
| Trompt      | 0 . 6605 ± 0 . 0153   | -                     |
| AutoInt     | 0 . 6840 ± 0 . 0126   | 0 . 6478 ± 0 . 0146   |
| MLP - Mixer | 0 . 6672 ± 0 . 0263   | 0 . 6294 ± 0 . 0200   |
| Excel ∗     | 0 . 6881 ± 0 . 0182   | 0 . 6664 ± 0 . 0179   |
| SAINT       | 0 . 6797 ± 0 . 0161   | -                     |
| FT - T      | 0 . 6787 ± 0 . 0149   | 0 . 6564 ± 0 . 0250   |
| T2G         | 0 . 6783 ± 0 . 0170   | 0 . 6570 ± 0 . 0273   |
| MLP ‡- lite | 0 . 6569 ± 0 . 0167   | 0 . 6328 ± 0 . 0155   |
| MLP ‡       | 0 . 6532 ± 0 . 0133   | 0 . 6336 ± 0 . 0140   |
| MLP †       | 0 . 6721 ± 0 . 0180   | 0 . 6463 ± 0 . 0262   |
| XGBoost     | 0 . 6039 ± 0 . 0134   | 0 . 6025 ± 0 . 0139   |
| LightGBM    | 0 . 6135 ± 0 . 0138   | 0 . 6122 ± 0 . 0144   |
| CatBoost    | 0 . 6088 ± 0 . 0132   | 0 . 6060 ± 0 . 0137   |
| TabR        | 0 . 6315 ± 0 . 0097   | 0 . 6197 ± 0 . 0096   |
| TabR ‡      | 0 . 6412 ± 0 . 0105   | -                     |
| MNCA        | 0 . 6154 ± 0 . 0083   | 0 . 6058 ± 0 . 0149   |
| MNCA ‡      | 0 . 6099 ± 0 . 0144   | 0 . 6028 ± 0 . 0157   |
| TabM ♠      | 0 . 6169 ± 0 . 0123   | 0 . 6131 ± 0 . 0126   |
| TabM        | 0 . 6328 ± 0 . 0172   | 0 . 6297 ± 0 . 0180   |
| TabM[G]     | 0 . 6369 ± 0 . 0179   | -                     |
| TabM mini   | 0 . 6314 ± 0 . 0142   | 0 . 6272 ± 0 . 0146   |
| TabM † mini | 0 . 6294 ± 0 . 0120   | 0 . 6241 ± 0 . 0118   |

kdd ipums la 97-small ↑

±

| Method      | Single model          | Ensemble                                |
|-------------|-----------------------|-----------------------------------------|
| MLP TabPFN  | 0 . 8828 ± 0 . 0061 - | 0 . 8845 ± 0 . 0055 0 . 8578 ± 0 . 0046 |
| ResNet      | 0 . 8823 ± 0 . 0070   | 0 . 8824 ± 0 . 0060                     |
| DCN2        | 0 . 8770 ± 0 . 0072   | 0 . 8824 ± 0 . 0068                     |
| SNN         | 0 . 8722 ± 0 . 0093   | 0 . 8733 ± 0 . 0083                     |
| Trompt      | 0 . 8847 ± 0 . 0070   | -                                       |
| AutoInt     | 0 . 8808 ± 0 . 0083   | 0 . 8830 ± 0 . 0081                     |
| MLP - Mixer | 0 . 8762 ± 0 . 0100   | 0 . 8770 ± 0 . 0088                     |
| Excel ∗     | 0 . 8803 ± 0 . 0054   | 0 . 8823 ± 0 . 0071                     |
| SAINT       | 0 . 8837 ± 0 . 0055   | -                                       |
| FT - T      | 0 . 8795 ± 0 . 0077   | 0 . 8792 ± 0 . 0062                     |
| T2G         | 0 . 8833 ± 0 . 0054   | 0 . 8841 ± 0 . 0062                     |
| MLP ‡- lite | 0 . 8765 ± 0 . 0108   | 0 . 8765 ± 0 . 0108                     |
| MLP ‡       | 0 . 8816 ± 0 . 0057   | 0 . 8818 ± 0 . 0048                     |
| MLP †       | 0 . 8757 ± 0 . 0101   | 0 . 8756 ± 0 . 0104                     |
| XGBoost     | 0 . 8825 ± 0 . 0089   | 0 . 8835 ± 0 . 0085                     |
| LightGBM    | 0 . 8792 ± 0 . 0075   | 0 . 8802 ± 0 . 0067                     |
| CatBoost    | 0 . 8793 ± 0 . 0088   | 0 . 8803 ± 0 . 0100                     |
| TabR        | 0 . 8798 ± 0 . 0081   | 0 . 8819 ± 0 . 0078                     |
| TabR ‡      | 0 . 8831 ± 0 . 0050   | -                                       |
| MNCA        | 0 . 8819 ± 0 . 0054   | 0 . 8832 ± 0 . 0048                     |
| MNCA ‡      | 0 . 8837 ± 0 . 0062   | 0 . 8860 ± 0 . 0059                     |
| TabM ♠      | 0 . 8845 ± 0 . 0063   | 0 . 8848 ± 0 . 0070                     |
| TabM        | 0 . 8823 ± 0 . 0079   | 0 . 8825 ± 0 . 0071                     |
| TabM[G]     | 0 . 8818 ± 0 . 0082   | -                                       |
| TabM mini   | 0 . 8784 ± 0 . 0123   | 0 . 8786 ± 0 . 0133                     |
| TabM † mini | 0 . 8779 0 . 0094     | 0 . 8784 0 . 0108                       |

±

isolet ↓

|                                                                                                                                                                                               | Single model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Ensemble                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Method MLP TabPFN ResNet DCN2 SNN Trompt AutoInt MLP - Mixer Excel ∗ SAINT FT - T T2G MLP ‡- lite MLP ‡ MLP † XGBoost LightGBM CatBoost TabR TabR ‡ MNCA MNCA ‡ TabM ♠ TabM TabM[G] TabM mini | 2 . 2744 ± 0 . 2203 - 2 . 2077 ± 0 . 2248 2 . 2449 ± 0 . 1579 2 . 4269 ± 0 . 2382 2 . 6219 ± 0 . 0315 2 . 6130 ± 0 . 1658 2 . 3344 ± 0 . 2073 2 . 8691 ± 0 . 0882 2 . 7696 ± 0 . 0200 2 . 4879 ± 0 . 2524 2 . 2867 ± 0 . 2489 2 . 2719 ± 0 . 1006 2 . 1832 ± 0 . 1124 2 . 0979 ± 0 . 1779 2 . 7567 ± 0 . 0470 2 . 7005 ± 0 . 0296 2 . 8847 ± 0 . 0227 1 . 9760 ± 0 . 1738 1 . 9919 ± 0 . 1813 1 . 7905 ± 0 . 1594 1 . 8912 ± 0 . 1851 1 . 8831 ± 0 . 1194 1 . 8433 ± 0 . 1196 1 . 9091 ± 0 . 1345 1 . 9421 ± 0 . 0971 | 2 . 0018 ± 0 . 1111 - 1 . 9206 ± 0 . 1478 2 . 0176 ± 0 . 0770 2 . 1142 ± 0 . 1262 - 2 . 3308 ± 0 . 1088 2 . 0915 ± 0 . 1159 2 . 5989 ± 0 . 0664 - 2 . 1501 ± 0 . 1506 1 . 9179 ± 0 . 1530 2 . 1026 ± 0 . 1088 2 . 0775 ± 0 . 0805 1 . 9283 ± 0 . 1334 2 . 7294 ± 0 . 0366 2 . 6903 ± 0 . 0290 2 . 8574 ± 0 . 0148 1 . 7627 ± 0 . 1520 - 1 . 6205 ± 0 . 1676 1 . 7147 ± 0 . 1348 1 . 8578 ± 0 . 1088 1 . 8230 ± 0 . 1197 - 1 . 9013 ± 0 . 0813 |

cpu act ↓

±

| Method          | Single model                            | Ensemble                                |
|-----------------|-----------------------------------------|-----------------------------------------|
| MLP TabPFN      | 2 . 6814 ± 0 . 2291 -                   | 2 . 4953 ± 0 . 1150 -                   |
| ResNet DCN2 SNN | 2 . 3933 ± 0 . 0641 2 . 7868 ± 0 . 1999 | 2 . 3005 ± 0 . 0397 2 . 4884 ± 0 . 0327 |
|                 | 2 . 5811 ± 0 . 1480                     | 2 . 3863 ± 0 . 0324                     |
| Trompt          | 2 . 2133 ± 0 . 0221                     | -                                       |
| AutoInt         | 2 . 2537 ± 0 . 0536                     | 2 . 1708 ± 0 . 0349                     |
| MLP - Mixer     | 2 . 3079 ± 0 . 0829                     | 2 . 1831 ± 0 . 0470                     |
| Excel ∗         | 2 . 3094 ± 0 . 2401                     | 2 . 1411 ± 0 . 0767                     |
| SAINT           | 2 . 2781 ± 0 . 0630                     | -                                       |
| FT - T          | 2 . 2394 ± 0 . 0508                     | 2 . 1494 ± 0 . 0268                     |
| T2G             | 2 . 2111 ± 0 . 0413                     | 2 . 1330 ± 0 . 0316                     |
| MLP ‡- lite     | 2 . 2730 ± 0 . 0457                     | 2 . 1899 ± 0 . 0419                     |
| MLP ‡           | 2 . 2671 ± 0 . 0383                     | 2 . 1940 ± 0 . 0433                     |
| MLP †           | 2 . 3309 ± 0 . 0719                     | 2 . 2516 ± 0 . 0574                     |
| XGBoost         | 2 . 5237 ± 0 . 3530                     | 2 . 4723 ± 0 . 3789                     |
| LightGBM        | 2 . 2223 ± 0 . 0894                     | 2 . 2067 ± 0 . 0916                     |
| CatBoost        | 2 . 1239 ± 0 . 0489                     | 2 . 1092 ± 0 . 0499                     |
| TabR            | 2 . 2980 ± 0 . 0529                     | 2 . 2228 ± 0 . 0501                     |
| TabR            | 2 . 1278 ± 0 . 0783                     | - 2 . 2339 ± 0 . 0508                   |
| MNCA            | 2 . 2603 ± 0 . 0479                     |                                         |
| MNCA ‡          | 2 . 2105 ± 0 . 0483                     | 2 . 1396 ± 0 . 0474                     |
| TabM ♠ TabM     | 2 . 1940 ± 0 . 0523 2 . 1402 0 . 0588   | 2 . 1677 ± 0 . 0487 2 . 1265 ± 0 . 0580 |
|                 | ±                                       |                                         |
| TabM[G]         | 2 . 1549 ± 0 . 0626                     | -                                       |
| TabM mini       | 2 . 1638 ± 0 . 0420                     | 2 . 1508 ± 0 . 0416                     |
| TabM † mini     | 2 . 1391 0 . 0542                       | 2 . 1221 0 . 0570                       |

## Brazilian houses ↓

Single model Ensemble

0

0179

.

0473

0

.

-

±

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

0

.

0

.

.

0

.

0

0

.

0

.

.

0

.

0

.

0

.

0

0

.

.

0

.

0

0

.

.

0

.

0

.

0

.

0505

0630

0477

0404

0513

0470

0450

0438

0479

0468

0426

0437

0421

0603

0541

0468

0490

0451

0527

0553

0443

0417

0424

0433

0416

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

.

0

0

.

0

.

0

.

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

0

.

0

.

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0181

0162

0172

0266

0234

0192

0156

0181

0205

0165

0180

0203

0209

0249

0270

0312

0152

0163

0157

0192

0213

0208

0201

0232

0215

0

.

0440

-

±

0

.

0

.

.

0

-

0

.

0

.

0

.

-

0

.

0

.

0

.

0

.

0

.

0

.

0

0

.

0

.

.

-

.

0

0

.

0

.

0

.

-

0

.

0

.

0458

0556

0427

0437

0484

0418

0412

0436

0397

0407

0409

0589

0535

0456

0454

0509

0511

0431

0413

0428

0406

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

Method

MLP

ResNet

TabPFN

DCN2

Trompt

SNN

AutoInt

-

Mixer

MLP

Excel

∗

SAINT

-

T2G

T

FT

‡-

MLP

‡

MLP

†

XGBoost

MLP

LightGBM

TabR

CatBoost

TabR

‡

MNCA

‡

MNCA

TabM

♠

TabM

TabM[G]

mini

TabM

TabM

†

mini

lite

±

0

.

0

.

0

.

0

.

0

0

.

.

0

.

0

0

.

.

0

.

0

.

0

.

0

0

.

0

.

0

0207

0207

0175

0207

0217

0262

0190

0204

0211

0206

0230

0226

0271

0287

.

.

0

0

0

0

0

0

0170

0332

.

0180

.

0191

.

.

0222

0233

.

0247

.

0230

Method

MLP

ResNet

TabPFN

DCN2

Trompt

SNN

AutoInt

-

MLP

Mixer

Excel

∗

SAINT

-

T2G

FT

T

‡-

MLP

‡

MLP

†

XGBoost

MLP

LightGBM

TabR

CatBoost

TabR

‡

MNCA

‡

MNCA

TabM

♠

TabM

TabM[G]

mini

TabM

TabM

†

mini

Method

MLP

ResNet

TabPFN

DCN2

Trompt

SNN

AutoInt

-

MLP

Mixer

Excel

∗

SAINT

-

T2G

FT

T

‡-

MLP

‡

MLP

†

XGBoost

MLP

LightGBM

TabR

CatBoost

TabR

‡

MNCA

‡

MNCA

TabM

♠

TabM

TabM[G]

mini

TabM

TabM

†

mini

lite

lite

## bank-marketing ↑

Single model Ensemble

0

0057

.

7860

0

.

-

±

0

.

0

0

.

0

.

.

0

.

0

.

0

.

0

0

.

0

.

0

.

.

0

0

.

0

.

.

0

.

0

0

.

.

0

0

.

.

0

.

0

0

.

.

0

0

.

.

0068

0076

0074

0071

0080

0059

0090

.

0

.

0

0

.

0

.

0

0

.

.

0

.

0

.

0058

.

0

0

.

0

0

0

0

0

0

0

0

0

0

0

0076

0058

.

0101

.

0092

.

0065

.

0078

.

0081

.

.

.

.

.

.

0054

0068

0088

0065

0081

0068

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

7859

7921

7836

7917

7975

7954

7957

7918

7953

7918

7947

7988

7981

8006

8013

8026

7995

8023

7961

7977

7908

7944

7935

0060

.

0

±

0

.

7887

0

.

.

7894

±

0

0

.

0

.

-

0

.

.

0

0

.

-

0

.

.

0

0

.

0

.

0

.

0

.

0

.

0

0

.

.

-

0

0

.

.

0

.

.

0

±

0064

.

0

±

0055

.

0

7941

-

0

.

0

±

.

0

7989

0

.

0086

## MagicTelescope ↑

Single model Ensemble

0

0060

.

8539

0

.

-

±

0

.

0

0

.

0

.

.

.

0

.

0

0

.

0

0

0

0

0

0

.

.

.

.

.

0

.

.

0

.

0

.

0

.

0

.

0

.

0

.

0

0

.

.

0

.

0

.

0

.

8589

8536

8432

8605

8571

8522

8480

8588

8595

8553

8591

8575

8593

8547

8550

8586

8682

8641

8602

8622

8607

8622

8600

8606

8644

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

0074

0068

0052

0056

0102

0080

0090

0046

0060

.

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0055

.

0061

.

0056

.

0054

.

0085

.

0094

.

.

.

.

.

.

.

.

.

.

0058

0070

0052

0061

0085

0058

0049

0055

0055

0088

0

±

.

8566

0

.

±

0

.

8579

0

.

0

.

-

0

.

.

0

0

.

-

0

.

.

0

0

.

0

.

0

.

0

.

0

.

0

0

.

.

-

0

.

0

.

0

.

0

.

-

0

.

0

8618

.

8673

8490

8651

8567

8560

8624

8543

8643

8595

8626

8605

8621

8556

8589

8588

8729

8628

8681

8622

8631

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

7943

.

8002

7917

7932

7882

7956

8001

7985

7951

7955

7977

8024

8008

8013

8030

8056

8015

8003

8010

7915

7944

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

.

0052

.

0066

.

0091

.

.

0054

0078

.

.

0048

0058

.

0106

.

.

0047

0071

.

.

.

.

.

.

.

.

.

.

.

.

.

0117

0093

0057

0072

0076

0082

0037

0077

0084

0068

0052

0045

0074

.

0061

.

0049

.

0064

.

.

0047

0046

.

.

0044

0034

.

0075

.

.

0051

0037

.

.

.

.

.

.

.

.

.

.

.

.

.

0044

0051

0037

0086

0110

0077

0038

0041

0064

0050

0046

0049

0075

Ailerons ↓

±

| Method      | Single model                            | Ensemble                                |
|-------------|-----------------------------------------|-----------------------------------------|
| MLP TabPFN  | 0 . 0002 ± 0 . 0000 -                   | 0 . 0002 ± 0 . 0000 -                   |
| ResNet DCN2 | 0 . 0002 ± 0 . 0000 0 . 0002 ± 0 . 0000 | 0 . 0002 ± 0 . 0000 0 . 0002 ± 0 . 0000 |
| SNN         | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| Trompt      | 0 . 0002 ± 0 . 0000                     | -                                       |
| AutoInt     | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| MLP - Mixer | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| Excel ∗     | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| SAINT       | 0 . 0002 ± 0 . 0000                     | -                                       |
| FT - T      | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| T2G         | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| MLP ‡- lite | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| MLP ‡       | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| MLP †       | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| XGBoost     | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| LightGBM    | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| CatBoost    | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| TabR        | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| TabR ‡      | 0 . 0002 ± 0 . 0000                     | -                                       |
| MNCA        | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| MNCA ‡      | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| TabM ♠      | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| TabM        | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| TabM[G]     | 0 . 0002 ± 0 . 0000                     | -                                       |
| TabM mini   | 0 . 0002 ± 0 . 0000                     | 0 . 0002 ± 0 . 0000                     |
| TabM † mini | 0 . 0002 0 . 0000                       | 0 . 0002 0 . 0000                       |

±

## OnlineNewsPopularity ↓

MiamiHousing2016 ↓

| Method      | Single model          | Ensemble              |
|-------------|-----------------------|-----------------------|
| MLP TabPFN  | 0 . 8643 ± 0 . 0007 - | 0 . 8632 ± 0 . 0005 - |
| ResNet      | 0 . 8665 ± 0 . 0011   | 0 . 8639 ± 0 . 0000   |
| DCN2        | 0 . 8714 ± 0 . 0013   | 0 . 8648 ± 0 . 0004   |
| SNN         | 0 . 8692 ± 0 . 0015   | 0 . 8665 ± 0 . 0005   |
| Trompt      | 0 . 8623 ± nan        | -                     |
| AutoInt     | 0 . 8636 ± 0 . 0022   | 0 . 8596 ± 0 . 0008   |
| MLP - Mixer | 0 . 8615 ± 0 . 0008   | 0 . 8598 ± 0 . 0004   |
| Excel ∗     | 0 . 8605 ± 0 . 0024   | 0 . 8556 ± nan        |
| SAINT       | 0 . 8600 ± 0 . 0007   | -                     |
| FT - T      | 0 . 8629 ± 0 . 0019   | 0 . 8603 ± 0 . 0000   |
| T2G         | 0 . 8632 ± 0 . 0009   | 0 . 8572 ± nan        |
| MLP ‡- lite | 0 . 8604 ± 0 . 0009   | 0 . 8591 ± 0 . 0004   |
| MLP ‡       | 0 . 8594 ± 0 . 0004   | 0 . 8585 ± 0 . 0001   |
| MLP †       | 0 . 8585 ± 0 . 0003   | 0 . 8581 ± 0 . 0001   |
| XGBoost     | 0 . 8545 ± 0 . 0002   | 0 . 8543 ± 0 . 0000   |
| LightGBM    | 0 . 8546 ± 0 . 0002   | 0 . 8544 ± 0 . 0000   |
| CatBoost    | 0 . 8532 ± 0 . 0003   | 0 . 8527 ± 0 . 0001   |
| TabR        | 0 . 8677 ± 0 . 0013   | 0 . 8633 ± 0 . 0009   |
| TabR ‡      | 0 . 8624 ± 0 . 0011   | -                     |
| MNCA        | 0 . 8651 ± 0 . 0003   | 0 . 8650 ± 0 . 0002   |
| MNCA ‡      | 0 . 8647 ± 0 . 0010   | 0 . 8624 ± 0 . 0006   |
| TabM ♠      | 0 . 8584 ± 0 . 0003   | 0 . 8581 ± 0 . 0001   |
| TabM        | 0 . 8579 ± 0 . 0003   | 0 . 8575 ± 0 . 0001   |
| TabM[G]     | 0 . 8579 ± 0 . 0004   | -                     |
| TabM mini   | 0 . 8588 ± 0 . 0004   | 0 . 8581 ± 0 . 0003   |
| TabM † mini | 0 . 8563 ± 0 . 0004   | 0 . 8558 ± 0 . 0002   |

±

| Method      | Single model          | Ensemble              |
|-------------|-----------------------|-----------------------|
| MLP TabPFN  | 0 . 1614 ± 0 . 0033 - | 0 . 1574 ± 0 . 0043 - |
| ResNet      | 0 . 1548 ± 0 . 0030   | 0 . 1511 ± 0 . 0027   |
| DCN2        | 0 . 1683 ± 0 . 0099   | 0 . 1575 ± 0 . 0047   |
| SNN         | 0 . 1618 ± 0 . 0029   | 0 . 1557 ± 0 . 0021   |
| Trompt      | 0 . 1478 ± 0 . 0028   | -                     |
| AutoInt     | 0 . 1537 ± 0 . 0035   | 0 . 1478 ± 0 . 0027   |
| MLP - Mixer | 0 . 1527 ± 0 . 0037   | 0 . 1479 ± 0 . 0033   |
| Excel ∗     | 0 . 1519 ± 0 . 0038   | 0 . 1442 ± 0 . 0022   |
| SAINT       | 0 . 1507 ± 0 . 0022   | -                     |
| FT - T      | 0 . 1514 ± 0 . 0029   | 0 . 1462 ± 0 . 0031   |
| T2G         | 0 . 1523 ± 0 . 0023   | 0 . 1478 ± 0 . 0024   |
| MLP ‡- lite | 0 . 1514 ± 0 . 0025   | 0 . 1479 ± 0 . 0017   |
| MLP ‡       | 0 . 1512 ± 0 . 0019   | 0 . 1470 ± 0 . 0024   |
| MLP †       | 0 . 1461 ± 0 . 0015   | 0 . 1433 ± 0 . 0022   |
| XGBoost     | 0 . 1440 ± 0 . 0029   | 0 . 1434 ± 0 . 0029   |
| LightGBM    | 0 . 1461 ± 0 . 0025   | 0 . 1455 ± 0 . 0030   |
| CatBoost    | 0 . 1417 ± 0 . 0021   | 0 . 1408 ± 0 . 0026   |
| TabR MNCA   | 0 . 1503 ± 0 . 0040   | - 0 . 0 . 0032        |
| ‡           | ± 0 . 1392 ± 0 . 0023 | ± 1477 ±              |
| MNCA ‡      | 0 . 1475 ± 0 . 0031   | 0 . 1438 ± 0 . 0024   |
| TabM ♠      | 0 . 1483 ± 0 . 0030   | 0 . 1465 ± 0 . 0029   |
| TabM        | 0 . 1478 ± 0 . 0012   | 0 . 1471 ± 0 . 0011   |
| TabM[G]     | 0 . 1482 ± 0 . 0012   | -                     |
| TabM mini   | 0 . 1481 ± 0 . 0021   | 0 . 1471 ± 0 . 0020   |
| TabM † mini | 0 . 1408 0 . 0019     | 0 . 1399 0 . 0018     |

±

credit ↑

| Method                                                                                                                                                                                        | Single                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Ensemble                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MLP TabPFN ResNet DCN2 SNN Trompt AutoInt MLP - Mixer Excel ∗ SAINT FT - T T2G MLP ‡- lite MLP ‡ MLP † XGBoost LightGBM CatBoost TabR TabR ‡ MNCA MNCA ‡ TabM ♠ TabM TabM[G] TabM mini TabM † | model 0 . 7735 ± 0 . 0042 - 0 . 7721 ± 0 . 0033 0 . 7703 ± 0 . 0034 0 . 7712 ± 0 . 0045 0 . 7740 ± 0 . 0006 0 . 7737 ± 0 . 0050 0 . 7748 ± 0 . 0038 0 . 7724 ± 0 . 0038 0 . 7739 ± 0 . 0052 0 . 7745 ± 0 . 0041 0 . 7744 ± 0 . 0046 0 . 7749 ± 0 . 0055 0 . 7734 ± 0 . 0034 0 . 7758 ± 0 . 0040 0 . 7698 ± 0 . 0027 0 . 7686 ± 0 . 0028 0 . 7734 ± 0 . 0035 0 . 7730 ± 0 . 0043 0 . 7723 ± 0 . 0037 0 . 7739 ± 0 . 0032 0 . 7734 ± 0 . 0045 0 . 7751 ± 0 . 0042 0 . 7760 ± 0 . 0043 0 . 7754 ± 0 . 0045 0 . 7752 ± 0 . 0047 0 . 7761 ± 0 . 0033 | 0 . 7729 ± 0 . 0047 0 . 7636 ± 0 . 0045 0 . 7738 ± 0 . 0027 0 . 7746 ± 0 . 0026 0 . 7716 ± 0 . 0059 - 0 . 7765 ± 0 . 0058 0 . 7768 ± 0 . 0059 0 . 7740 ± 0 . 0069 - 0 . 7767 ± 0 . 0040 0 . 7762 ± 0 . 0057 0 . 7767 ± 0 . 0075 0 . 7747 ± 0 . 0043 0 . 7772 ± 0 . 0055 0 . 7706 ± 0 . 0029 0 . 7726 ± 0 . 0034 0 . 7752 ± 0 . 0038 0 . 7740 ± 0 . 0040 - 0 . 7757 ± 0 . 0026 0 . 7754 ± 0 . 0040 0 . 7755 ± 0 . 0049 0 . 7771 ± 0 . 0044 - 0 . 7754 ± 0 . 0048 0 . 7760 ± 0 . 0028 |

elevators ↓

±

| Method          | Single model                            | Ensemble                           |
|-----------------|-----------------------------------------|------------------------------------|
| MLP TabPFN      | 0 . 0020 ± 0 . 0001 -                   | 0 . 0019 ± 0 . 0000 -              |
| ResNet DCN2 SNN | 0 . 0019 ± 0 . 0000 0 . 0019 ± 0 . 0000 | 0 . 0019 ± 0 . 0000 0 . 0019 ± 0 . |
|                 |                                         | 0000                               |
|                 | 0 . 0020 ± 0 . 0001                     | 0 . 0019 ± 0 . 0000                |
| Trompt          | 0 . 0018 ± 0 . 0000                     | -                                  |
| AutoInt         | 0 . 0019 ± 0 . 0000                     | 0 . 0018 ± 0 . 0000                |
| MLP - Mixer     | 0 . 0019 ± 0 . 0000                     | 0 . 0018 ± 0 . 0000                |
| Excel ∗         | 0 . 0019 ± 0 . 0000                     | 0 . 0018 ± 0 . 0000                |
| SAINT           | 0 . 0018 ± 0 . 0000                     | -                                  |
| FT - T          | 0 . 0019 ± 0 . 0000                     | 0 . 0018 ± 0 . 0000                |
| T2G             | 0 . 0019 ± 0 . 0000                     | 0 . 0018 ± 0 . 0000                |
| MLP ‡- lite     | 0 . 0019 ± 0 . 0000                     | 0 . 0018 ± 0 . 0000                |
| MLP ‡           | 0 . 0018 ± 0 . 0000                     | 0 . 0018 ± 0 . 0000                |
| MLP †           | 0 . 0018 ± 0 . 0000                     | 0 . 0018 ± 0 . 0000                |
| XGBoost         | 0 . 0020 ± 0 . 0000                     | 0 . 0020 ± 0 . 0000                |
| LightGBM        | 0 . 0020 ± 0 . 0000                     | 0 . 0020 ± 0 . 0000                |
| CatBoost        | 0 . 0020 ± 0 . 0000                     | 0 . 0019 ± 0 . 0000                |
| TabR            | 0 . 0049 ± 0 . 0000                     | 0 . 0049 ± 0 . 0000                |
| TabR ‡          | 0 . 0019 ± 0 . 0001                     |                                    |
| MNCA            | 0 . 0019 ± 0 . 0000                     | - 0 . 0019 ± 0 . 0000              |
| MNCA ‡          | 0 . 0018 ± 0 . 0000                     | 0 . 0018 ± 0 . 0000                |
| TabM ♠          | 0 . 0019 ± 0 . 0000                     | 0 . 0018 ± 0 . 0000                |
| TabM            | 0 . 0018 ± 0 . 0000                     | 0 . 0018 ± 0 . 0000                |
| TabM[G]         | 0 . 0018 ± 0 . 0000                     | -                                  |
| TabM mini       | 0 . 0018 ± 0 . 0000                     | 0 . 0018 ± 0 . 0000                |
| TabM † mini     | 0 . 0018 0 . 0000                       | 0 . 0018 0 . 0000                  |

## house sales ↓

Single model

0

.

1790

0

.

0009

-

±

0

.

0

.

0

.

0

.

0

0

.

.

0

.

0

0

.

0

.

.

0

.

0

0

.

0

.

.

0

.

0

0

.

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

1755

1800

1862

1667

1704

1700

1713

1690

1713

1689

1699

1690

1687

1692

1694

1669

1689

1636

1737

1694

1692

1666

1667

1673

1652

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

.

0

.

0

.

0032

0014

0008

nan

0

.

0007

0

.

0

.

0

0

.

0

.

.

0

.

0

.

0

.

0

0

.

0

.

.

0

.

0

0014

0010

0010

0015

0010

0008

0005

0004

0004

0003

0001

0009

.

0

.

0

0

0

0

0

0

0009

0013

.

0007

.

0011

.

0003

.

0003

.

0004

.

0003

Method

MLP

ResNet

TabPFN

DCN2

Trompt

SNN

AutoInt

-

Mixer

MLP

Excel

∗

SAINT

-

T2G

T

FT

‡-

MLP

‡

MLP

†

XGBoost

MLP

LightGBM

TabR

CatBoost

TabR

‡

MNCA

‡

MNCA

TabM

♠

TabM

TabM[G]

mini

TabM

TabM

†

mini

lite

±

Ensemble

0

.

1763

-

±

0

.

0

0

.

.

-

0

0

.

.

0

.

-

0

.

.

0

0

.

0

.

0

.

0

.

.

0

0

.

0

.

-

.

0

0

.

0

0

.

.

-

0

.

0

.

1738

1770

1778

1670

1690

1668

1659

1664

1687

1676

1681

1686

1689

1667

1657

1714

1670

1680

1662

1668

1644

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

.

0

.

0

0

.

.

0003

0006

0004

0015

0

0

.

.

0008

0005

nan

0

nan

.

0004

0

.

0007

0

0

0

0

0

0

0

0

0

0

0

0

.

0003

.

0001

.

0001

.

0001

.

.

0003

0000

.

0005

.

.

.

.

.

0003

0005

0002

0001

0001

Method

MLP

ResNet

TabPFN

DCN2

Trompt

SNN

AutoInt

-

MLP

Mixer

Excel

∗

SAINT

-

T2G

FT

T

‡-

MLP

‡

MLP

†

XGBoost

MLP

LightGBM

TabR

CatBoost

TabR

‡

MNCA

‡

MNCA

TabM

♠

TabM

TabM[G]

mini

TabM

TabM

†

mini

Method

MLP

ResNet

TabPFN

DCN2

Trompt

SNN

AutoInt

-

∗

MLP

Mixer

Excel

SAINT

-

T2G

FT

T

‡-

MLP

‡

MLP

†

XGBoost

MLP

LightGBM

TabR

CatBoost

TabR

‡

MNCA

‡

MNCA

TabM

♠

TabM

TabM[G]

mini

TabM

TabM

†

mini

lite

lite Single model

fifa ↓

0

.

8038

0

.

0124

-

±

0

.

0

0

.

0

.

0

.

0

.

.

0

.

0

0

.

0

.

0

.

.

0

0

.

0

.

.

0

.

0

0

.

.

0

0

.

.

0

.

0

0

.

.

0

0

.

.

Ensemble

0

.

8011

-

±

0

.

0

.

.

0

-

0

.

.

0

0

.

-

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

0

.

.

-

0

0

.

.

0

0

.

.

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

-

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

0

.

0

.

0135

0140

0140

0128

0180

0119

0111

0132

0118

.

0

0

0

0

0

0

0

0

0

0

0

±

0

0139

.

0118

.

0092

.

0104

.

.

0108

.

0120

.

.

.

.

.

.

0119

0116

0136

0138

0107

0144

0135

8025

8074

8046

7880

7936

7923

7909

7928

7901

7928

7940

7907

7806

7806

7800

7835

7902

7914

7967

7909

7974

7953

7948

0

0

.

.

0156

0135

7938

0

±

.

7771

±

0

.

0107

0

## medical charges ↓

Single model Ensemble

0

0001

.

0816

0

.

-

±

0

.

0

0

.

0

.

.

0

0

.

.

0

.

0

0

0

0

.

.

.

.

0

0

.

0

.

0

.

0

.

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0824

0827

0818

0812

0814

0822

0817

0814

0814

0813

0812

0812

0812

0820

0825

0816

0815

0811

0811

0809

0813

0812

0812

0813

0811

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

.

0

0003

0003

0006

0

.

nan

.

0

0002

0

.

.

0

0007

.

0

0

.

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0002

0004

.

0002

.

0002

.

0002

.

0001

.

.

.

.

.

.

.

.

.

.

.

.

.

0000

0000

0001

0000

0002

0001

0001

0000

0001

0000

0000

0000

0001

0

0814

.

-

±

0

.

0

.

0

.

-

0

.

.

0

0

.

-

0

.

.

0

0

.

0

.

0

0

.

0

.

0

.

0

.

.

-

0

0

.

.

0

0

.

.

-

0

.

0

0813

.

0811

0817

0817

0815

0814

0811

0813

0812

0811

0810

0809

0811

0820

0825

0815

0812

0810

0808

0812

0812

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

7920

.

7761

7985

8031

7993

7886

7903

7862

7888

7904

7898

7870

7800

7787

7795

7817

7863

7933

7866

7954

7942

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

.

0143

.

0149

.

0147

.

0129

.

.

0133

0127

.

0161

.

.

0183

0130

.

.

.

.

.

.

.

.

.

.

.

.

.

0141

0096

0114

0122

0114

0114

0120

0145

0106

0160

0148

0176

0117

.

0000

.

0001

.

0001

0

.

0001

0

0

.

0000

.

0001

nan

0

nan

.

0000

0

.

0000

0

0

0

0

0

0

0

0

0

0

0

0

.

.

.

.

.

.

.

.

.

.

.

.

0001

0000

0000

0000

0000

0000

0000

0000

0000

0000

0000

0000

Method

MLP

ResNet

TabPFN

DCN2

Trompt

SNN

AutoInt

-

Mixer

MLP

Excel

∗

SAINT

-

T2G

T

FT

‡-

MLP

‡

MLP

†

XGBoost

MLP

LightGBM

TabR

CatBoost

TabR

‡

MNCA

‡

MNCA

TabM

♠

TabM

TabM[G]

mini

TabM

TabM

†

mini

Method

MLP

ResNet

TabPFN

DCN2

Trompt

SNN

AutoInt

-

Mixer

MLP

Excel

∗

SAINT

-

T2G

T

FT

‡-

MLP

‡

MLP

†

XGBoost

MLP

LightGBM

TabR

CatBoost

TabR

‡

MNCA

‡

MNCA

TabM

♠

TabM

TabM[G]

mini

TabM

TabM

†

mini

lite

lite Single model

pol ↓

5

.

5244

0

.

5768

-

±

6

.

6

.

6

.

3

.

3

3

.

.

3

.

2

.

2

.

2

.

2

.

2

2

.

4

.

4

.

3

.

6

.

.

2

5

.

.

2

.

3

3

.

.

3

3

2

.

.

.

3739

1816

5374

2337

2011

3295

0682

6974

7203

9539

8239

5452

4958

2320

2963

6320

0708

5770

7878

9083

3595

0198

0358

1351

0

.

0

.

0

.

0

.

0

0

.

.

0

.

0

0

.

0

.

.

0

.

0

.

0

.

0

.

0

.

0

0

.

.

0

0

.

.

0

.

0

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

±

.

6286

7366

9479

0605

2921

3379

2389

1666

1858

1994

2173

1221

1292

3369

0644

1006

5368

1689

4884

1364

4017

2975

.

±

0

.

1952

0

.

3077

2808

±

0

.

0343

jannis ↑

Single model

0

0018

.

7840

0

.

-

±

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

0

.

0

.

.

0

.

0

0

.

0

.

.

0

.

0

.

0

.

0

0

.

.

0

.

0

.

0

.

0

.

0

.

0

.

7923

7818

7712

8027

7927

7933

7954

7940

7971

7998

7923

7947

7891

7956

7967

7985

7983

8051

7993

8068

8066

8080

8064

8053

8078

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

.

0

.

0

.

0029

0024

0025

nan

0

.

0025

0

.

0

.

0

0

.

0

.

.

0

.

0

.

0

.

0

.

0

.

0

0

.

.

0

0

0018

0015

0028

0028

0024

0018

0017

0013

0017

0019

0018

0022

.

.

0

0

0

0

0

0

0019

0023

.

0021

.

0015

.

0019

.

0018

.

0012

.

0008

Ensemble

4

.

9945

-

±

5

.

5

.

.

5

-

2

.

.

2

2

.

-

2

.

2

.

2

.

2

.

2

.

4

.

.

4

3

.

5

.

-

.

5

2

.

3

.

2

.

-

3

.

8181

5959

1814

7999

8698

5816

3718

6282

5266

3700

3651

1880

2548

5505

5578

3773

6717

2130

9595

0478

2

±

.

2383

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

Ensemble

0

±

.

7872

0

.

±

0

.

7419

0

0

.

.

-

0

.

.

0

0

.

-

0

.

.

0

0

.

0

.

0

.

0

.

.

0

0

.

0

.

-

0

.

0

.

0

.

0

.

-

0

.

0

.

7825

7958

7859

7983

8019

8021

7998

8052

7945

7967

7900

7968

7998

8009

8023

8042

8128

8075

8102

8066

8086

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

.

0

.

0

.

.

0

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

0

.

0

5923

6054

8243

7775

1776

2577

0368

0724

0730

0605

0867

1223

3110

0488

.

0

0

0

0

0

0

.

4036

0896

.

5463

.

0530

.

3107

.

3979

.

2061

.

0

0111

.

0

0018

0007

.

.

0

0

0010

0011

0009

.

0

.

0

0

0012

.

0013

.

nan

0

nan

0006

.

0

0010

.

0

0

0

0

0

0

0

0

0

0

0

0

.

0011

.

0006

.

0005

0007

.

.

0018

.

0012

.

0013

.

.

.

.

.

0007

0004

0017

0001

0005

Method

MLP

ResNet

TabPFN

DCN2

Trompt

SNN

AutoInt

-

∗

Mixer

MLP

Excel

SAINT

-

T2G

T

FT

‡-

MLP

‡

MLP

†

XGBoost

MLP

LightGBM

TabR

CatBoost

TabR

‡

MNCA

‡

MNCA

TabM

♠

TabM

TabM[G]

mini

TabM

TabM

†

mini

superconduct ↓

±

MiniBooNE ↑

| Method      | Single model           | Ensemble                                  |
|-------------|------------------------|-------------------------------------------|
| MLP TabPFN  | 10 . 8740 ± 0 . 0868 - | 10 . 4118 ± 0 . 0429 -                    |
| ResNet DCN2 | 10 . 7711 ± 0 . 1454   | 10 . 3495 ± 0 . 0168 10 . 4342 ± 0 . 0179 |
|             | 10 . 8108 ± 0 . 0957   |                                           |
| SNN         | 10 . 8562 ± 0 . 1300   | 10 . 3342 ± 0 . 0509                      |
| Trompt      | 10 . 4442 ± nan        | -                                         |
| AutoInt     | 11 . 0019 ± 0 . 1391   | 10 . 4469 ± 0 . 0521                      |
| MLP - Mixer | 10 . 7502 ± 0 . 0800   | 10 . 3281 ± 0 . 0450                      |
| Excel ∗     | 11 . 0879 ± 0 . 1571   | 10 . 4094 ± nan                           |
| SAINT       | 10 . 7807 ± 0 . 1074   | -                                         |
| FT - T      | 10 . 8256 ± 0 . 1692   | 10 . 3391 ± 0 . 0794                      |
| T2G         | 10 . 8310 ± 0 . 1406   | 10 . 3017 ± nan                           |
| MLP ‡- lite | 10 . 5058 ± 0 . 0758   | 10 . 2322 ± 0 . 0463                      |
| MLP ‡       | 10 . 5061 ± 0 . 0330   | 10 . 2440 ± 0 . 0127                      |
| MLP †       | 10 . 7220 ± 0 . 0757   | 10 . 3758 ± 0 . 0606                      |
| XGBoost     | 10 . 1610 ± 0 . 0201   | 10 . 1413 ± 0 . 0025                      |
| LightGBM    | 10 . 1634 ± 0 . 0118   | 10 . 1552 ± 0 . 0050                      |
| CatBoost    | 10 . 2422 ± 0 . 0222   | 10 . 2116 ± 0 . 0058                      |
| TabR        | 10 . 8842 ± 0 . 1073   | 10 . 4800 ± 0 . 0280                      |
| TabR ‡      | 10 . 3835 ± 0 . 0562   | -                                         |
| MNCA        | 10 . 4419 ± 0 . 0640   | 10 . 2926 ± 0 . 0261                      |
| MNCA ‡      | 10 . 5651 ± 0 . 0616   | 10 . 3155 ± 0 . 0253                      |
| TabM ♠      | 10 . 3379 ± 0 . 0338   | 10 . 1943 ± 0 . 0291                      |
| TabM        | 10 . 2628 ± 0 . 0275   | 10 . 2300 ± 0 . 0108                      |
| TabM[G]     | 10 . 2572 ± 0 . 0463   | -                                         |
| TabM mini   | 10 . 2472 ± 0 . 0208   | 10 . 2094 ± 0 . 0057                      |
| TabM † mini | 10 . 1326 0 . 0186     | 10 . 0866 0 . 0070                        |

Single model

0

.

9480

0

.

0007

-

±

0

.

0

.

0

.

0

.

0

.

0

.

0

.

0

0

.

0

0

0

0

.

.

.

.

0

.

.

0

.

0

.

0

.

0

0

.

.

0

.

0

0

.

.

0

.

0

.

0

.

9488

9476

9433

9473

9446

9447

9430

9467

9471

9475

9466

9473

9482

9422

9436

9453

9487

9475

9488

9493

9500

9503

9496

9495

9490

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

.

0

0

.

0011

0011

0013

.

0

.

nan

0

.

0014

0

0014

.

0

0

.

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0

0009

0015

.

.

0014

0014

.

0009

.

0010

.

.

.

.

.

.

.

.

.

.

.

.

.

0008

0009

0006

0008

0008

0007

0010

0012

0005

0006

0010

0005

0004

lite Ensemble

0

±

.

9498

0

.

±

0

.

9266

0

.

0

.

-

0

.

.

0

0

.

-

0

.

.

0

0

.

0

.

0

0

.

0

.

0

.

0

.

.

-

0

0

.

.

0

0

.

.

-

0

.

0

9500

.

9492

9470

9504

9491

9473

9483

9451

9486

9508

9478

9493

9492

9427

9452

9459

9500

9505

9501

9505

9501

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

±

0

.

0001

0

0

.

0012

0

0

.

0005

.

0010

.

0010

0

0

.

0002

.

0010

nan

0

nan

.

0010

0

.

0004

0

0

0

0

0

0

0

0

0

0

0

0

.

.

.

.

.

.

.

.

.

.

.

.

0004

0001

0003

0003

0005

0002

0001

0008

0002

0002

0002

0002

↓

|             | nyc-taxi-green-dec-2016                 | nyc-taxi-green-dec-2016   | particulate-matter-ukair-2017   | particulate-matter-ukair-2017         | particulate-matter-ukair-2017   |
|-------------|-----------------------------------------|---------------------------|---------------------------------|---------------------------------------|---------------------------------|
| Method      | Single model                            | Ensemble                  | Method                          | Single model                          | Ensemble                        |
| MLP         | 0 . 3951 ± 0 . 0009                     | 0 . 3921 ± 0 . 0003       | MLP                             | 0 . 3759 ± 0 . 0004                   | 0 . 3729 ± 0 . 0003             |
| TabPFN      | - 0 . 3899 ± 0 . 0016                   | - 0 . 3873 0 . 0009       | TabPFN ResNet                   | - 0 . 3743 0 . 0007                   | - 0 . 3718 ± 0 . 0005           |
| ResNet DCN2 | 0 . 3919 0 . 0009                       | ± 0 . 3889 0 . 0003       | DCN2                            | ± 0 . 3759 0 . 0012                   | 0 . 3738 0 . 0004               |
| SNN         | ± 0 . 3933 0 . 0013                     | ± 0 . 3899 0 . 0004       | SNN                             | ± 0 . 3790 ± 0 . 0007                 | ± 0 . 3744 ± 0 . 0002           |
| Trompt      | ± 0 . 3979 ± nan                        | ± -                       | Trompt                          | 0 . 3700 ± nan                        | -                               |
| AutoInt     | 0 . 4084 ± 0 . 0256                     | 0 . 3967 ± 0 . 0059       | AutoInt                         | 0 . 3723 ± 0 . 0011                   | 0 . 3692 ± 0 . 0010             |
| MLP - Mixer | 0 . 3914 ± 0 . 0026                     | 0 . 3861 ± 0 . 0013       | MLP - Mixer                     | 0 . 3741 ± 0 . 0010                   | 0 . 3698 ± 0 . 0004             |
| Excel ∗     | 0 . 3969 ± 0 . 0036                     | 0 . 3897 ± nan            | Excel ∗                         | 0 . 3699 ± 0 . 0014                   | 0 . 3652 ± nan                  |
| SAINT       | 0 . 3905 ± 0 . 0013                     | -                         | SAINT                           | 0 . 3704 ± 0 . 0014                   | -                               |
| FT - T      | 0 . 3937 ± 0 . 0064                     | 0 . 3889 ± 0 . 0018       | FT - T                          | 0 . 3735 ± 0 . 0012                   | 0 . 3686 ± 0 . 0004             |
| T2G         | 0 . 3908 ± 0 . 0045                     | 0 . 3858 ± nan            | T2G                             | 0 . 3676 ± 0 . 0024                   | 0 . 3631 ± nan                  |
| MLP ‡- lite | 0 . 3812 ± 0 . 0018                     | 0 . 3761 ± 0 . 0016       | MLP ‡- lite                     | 0 . 3665 ± 0 . 0008                   | 0 . 3642 ± 0 . 0003             |
| MLP ‡       | 0 . 3795 ± 0 . 0016                     | 0 . 3733 ± 0 . 0013       | MLP ‡                           | 0 . 3657 ± 0 . 0007                   | 0 . 3629 ± 0 . 0002             |
| MLP †       | 0 . 3680 0 . 0006                       | 0 . 3653 0 . 0005         | MLP †                           | 0 . 3649 0 . 0011                     | 0 . 3637 0 . 0008               |
| XGBoost     | ± 0 . 3792 0 . 0002                     | ± 0 . 3787 0 . 0000       | XGBoost                         | ± 0 . 3641 ± 0 . 0001                 | ± 0 . 3640 ± 0 . 0000           |
| LightGBM    | ± 0 . 3688 ± 0 . 0002                   | ± 0 . 3684 ± 0 . 0000     | LightGBM                        | 0 . 3637 ± 0 . 0001                   | 0 . 3635 ± 0 . 0000             |
| CatBoost    | 0 . 3647 ± 0 . 0005                     | 0 . 3632 ± 0 . 0003       | CatBoost                        | 0 . 3647 ± 0 . 0004                   | 0 . 3637 ± 0 . 0002             |
| TabR        | 0 . 3577 ± 0 . 0222                     | 0 . 3380 ± 0 . 0027       | TabR                            | 0 . 3613 ± 0 . 0005                   | 0 . 3590 ± 0 . 0002             |
| TabR ‡      | 0 . 3725 ± 0 . 0091                     | -                         | TabR ‡                          | 0 . 3596 ± 0 . 0004                   | -                               |
| MNCA        | 0 . 3728 ± 0 . 0012                     | 0 . 3720 ± 0 . 0010       | MNCA                            | 0 . 3670 ± 0 . 0004                   | 0 . 3649 ± 0 . 0002             |
| MNCA ‡      | 0 . 3536 ± 0 . 0052                     | 0 . 3407 ± 0 . 0009       | MNCA ‡                          | 0 . 3646 ± 0 . 0001                   | 0 . 3643 ± 0 . 0000             |
| TabM ♠      | 0 . 3866 ± 0 . 0006                     | 0 . 3855 ± 0 . 0003       | TabM ♠                          | 0 . 3686 ± 0 . 0006                   | 0 . 3679 ± 0 . 0003             |
| TabM        | 0 . 3849 ± 0 . 0005                     | 0 . 3843 ± 0 . 0002       | TabM                            | 0 . 3671 ± 0 . 0007                   | 0 . 3665 ± 0 . 0002             |
| TabM[G]     | 0 . 3848 ± 0 . 0005                     | -                         | TabM[G]                         | 0 . 3667 ± 0 . 0009                   | -                               |
| TabM mini   | 0 . 3853 ± 0 . 0005                     | 0 . 3845 ± 0 . 0003       | TabM mini                       | 0 . 3664 ± 0 . 0006                   | 0 . 3655 ± 0 . 0002             |
| TabM † mini | 0 . 3485 ± 0 . 0038                     | 0 . 3448 ± 0 . 0020       | TabM † mini                     | 0 . 3593 ± 0 . 0004                   | 0 . 3589 ± 0 . 0000             |
|             | road-safety ↑                           | road-safety ↑             | year ↓                          | year ↓                                | year ↓                          |
| Method      | Single model                            | Ensemble                  | Method                          | Single model                          | Ensemble                        |
| MLP         | 0 . 7857 0 . 0019                       | 0 . 7873 0 . 0004         | MLP                             | 8 . 9628 0 . 0232                     | 8 . 8931 ± 0 . 0066             |
| TabPFN      | ± -                                     | ± 0 . 7338 ± 0 . 0032     | TabPFN                          | ± -                                   | -                               |
| ResNet      | 0 . 7875 ± 0 . 0007                     | 0 . 7898 ± 0 . 0008       | ResNet                          | 8 . 9658 ± 0 . 0239                   | 8 . 8755 ± 0 . 0066             |
| DCN2        | 0 . 7781 ± 0 . 0014                     | 0 . 7823 ± 0 . 0012       | DCN2                            | 9 . 2761 ± 0 . 0401                   | 9 . 0640 ± 0 . 0156             |
| SNN         | 0 . 7847 ± 0 . 0010                     | 0 . 7865 ± 0 . 0002       | SNN                             | 9 . 0054 ± 0 . 0256                   | 8 . 9351 ± 0 . 0073             |
| Trompt      | 0 . 7804 ± nan                          | -                         | Trompt                          | 8 . 9707 ± nan                        | -                               |
| AutoInt     | 0 . 7826 ± 0 . 0030                     | 0 . 7883 ± 0 . 0013       | AutoInt                         | 9 . 0430 ± 0 . 0280                   | 8 . 9619 ± 0 . 0092             |
| MLP - Mixer | 0 . 7878 ± 0 . 0032                     | 0 . 7919 ± 0 . 0015       | MLP - Mixer                     | 8 . 9589 ± 0 . 0182                   | 8 . 9086 ± 0 . 0177             |
| Excel ∗     | 0 . 7864 0 . 0053                       | 0 . 7907 ± nan            | Excel ∗                         | 9 . 0395 ± 0 . 0266                   | 8 . 9551 ± nan                  |
| SAINT       | ± 0 . 7584 ± 0 . 0584                   | -                         | SAINT                           | 9 . 0248 ± 0 . 0225                   | -                               |
| FT - T      | 0 . 7907 ± 0 . 0012                     | 0 . 7943 ± 0 . 0007       | FT - T                          | 9 . 0005 ± 0 . 0215                   | 8 . 9360 ± 0 . 0013             |
| T2G         | 0 . 7912 ± 0 . 0026                     | 0 . 7961 ± nan            | T2G                             | 8 . 9775 ± 0 . 0138                   | 8 . 8979 ± nan                  |
| MLP ‡- lite | 0 . 7867 ± 0 . 0018                     | 0 . 7903 ± 0 . 0002       | MLP ‡- lite                     | 8 . 9355 ± 0 . 0103                   | 8 . 9063 ± 0 . 0030             |
| MLP ‡       | 0 . 7853 ± 0 . 0014                     | 0 . 7881 ± 0 . 0007       | MLP ‡                           | 8 . 9455 ± 0 . 0173                   | 8 . 9083 ± 0 . 0046             |
| MLP †       | 0 . 7899 0 . 0009                       | 0 . 7935 ± 0 . 0003       | MLP †                           | 8 . 9379 0 . 0206                     | 8 . 8753 ± 0 . 0038             |
| XGBoost     | ± 0 . 8101 ± 0 . 0017                   | 0 . 8129 ± 0 . 0004       | XGBoost                         | ± 9 . 0307 ± 0 . 0028                 | 9 . 0245 0 . 0015               |
| LightGBM    | 0 . 7982 ± 0 . 0012                     | 0 . 7996 ± 0 . 0005       | LightGBM                        | 9 . 0200 ± 0 . 0025                   | ± 9 . 0128 ± 0 . 0015           |
| CatBoost    | 0 . 8012 ± 0 . 0009                     | 0 . 8022 ± 0 . 0002       | CatBoost                        | 9 . 0370 ± 0 . 0073                   | 9 . 0054 ± 0 . 0028             |
|             | 0 . 8403 0 . 0014                       | 0 . 8441 0 . 0005         |                                 | 9 . 0069 0 . 0152                     |                                 |
| TabR        | ±                                       | ±                         | TabR                            | ±                                     | 8 . 9132 ± 0 . 0088             |
| TabR ‡      | 0 . 8374 ± 0 . 0013                     | - 0 . 8121 0 . 0006       | TabR ‡                          | 8 . 9721 ± 0 . 0105 8 . 9476 0 . 0152 | - 8 . 8977 0 . 0037             |
| MNCA MNCA ‡ | 0 . 8080 ± 0 . 0013 0 . 8232 ± 0 . 0017 | ± 0 . 8287 ± 0 . 0008     | MNCA MNCA ‡                     | ± 8 . 8973 ± 0 . 0082                 | ± 8 . 8550 ± 0 . 0031           |
| TabM ♠      | 0 . 7946 ± 0 . 0013                     | 0 . 7961 ± 0 . 0005       | TabM ♠                          | 8 . 8701 ± 0 . 0110                   | 8 . 8517 0 . 0022               |
| TabM        | 0 . 7958 ± 0 . 0011                     | 0 . 7968 ± 0 . 0004       | TabM                            | 8 . 8705 ± 0 . 0043                   | ± 8 . 8642 ± 0 . 0028           |
| TabM[G]     | 0 . 7954 ± 0 . 0016                     | -                         | TabM[G]                         | 8 . 8723 ± 0 . 0080                   | -                               |
| TabM mini   | 0 . 7933 ± 0 . 0030                     | 0 . 7970 ± 0 . 0006       | TabM mini                       | 8 . 9164 ± 0 . 0089                   | 8 . 9021 ± 0 . 0036             |
| TabM † mini | 0 . 7999 ± 0 . 0023                     | 0 . 8059 ± 0 . 0012       | TabM † mini                     | 8 . 8737 ± 0 . 0119                   | 8 . 8564 ± 0 . 0054             |

↓

Table 20: Extended results for TabReD Rubachev et al. (2024) benchmark. Results are grouped by datasets. One ensemble consists of five models trained independently under different random seeds.

sberbank-housing ↓

±

|                                                                                                                                                                 | Single model                                                                                                                                          | Ensemble                                                                                                          |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| Method MLP TabPFN ResNet DCN2 SNN Trompt AutoInt MLP - Mixer Excel ∗ SAINT FT - T T2G MLP ‡- lite MLP ‡ MLP † XGBoost LightGBM CatBoost TabR TabR ‡ MNCA MNCA ‡ | 0 . 2529 ± 0 . 0078 - - 0 . 2616 ± 0 . 0049 0 . 2671 ± 0 . 0140 0 . 2509 ± nan                                                                        | 0 . 2474 ± 0 . 0052 - - 0 . 2506 ± 0 . 0015 0 . 2555 0 . 0033 0 . 2485 ± nan - 0 . 2367 ± 0 . 0010 0 . 2343 ± nan |
|                                                                                                                                                                 | - -                                                                                                                                                   | ± - - -                                                                                                           |
|                                                                                                                                                                 | 0 . 2383 0 .                                                                                                                                          | 0 . 2327 ± 0 . 0 . 2416 ± 0 . 0 . 2467 ± 0 . 0002 0 . 2473 ± 0 . 0016                                             |
|                                                                                                                                                                 | ±                                                                                                                                                     |                                                                                                                   |
|                                                                                                                                                                 | 0 . 2533 ± 0 . 0046 0 . 2467 ± 0 . 0019 0 . 2440 ± 0 . 0 . 2416 ± 0 . 0 . 2528 ± 0 . 0 . 2412 ± 0 . 0 . 2419 ± 0 . 0 . 2468 ± 0 . 0 . 2482 ± 0 . 0034 | 0 . 2503 ± 0 . 0029 0007                                                                                          |
|                                                                                                                                                                 | 0038 0025 0055 0031                                                                                                                                   | 0 . 2355 ± 0 . 0006                                                                                               |
|                                                                                                                                                                 | 0032 0012                                                                                                                                             |                                                                                                                   |
|                                                                                                                                                                 |                                                                                                                                                       | 0009                                                                                                              |
|                                                                                                                                                                 | 0009                                                                                                                                                  |                                                                                                                   |
|                                                                                                                                                                 | 0 . 2820 ± 0 . 0323                                                                                                                                   | 0 . 2603 ± 0 . 0048                                                                                               |
|                                                                                                                                                                 | 0 . 2542 ± 0 . 0101                                                                                                                                   | -                                                                                                                 |
|                                                                                                                                                                 | 0 . 2593 ± 0 . 0053                                                                                                                                   | 0 . 2520 ± 0 . 0032                                                                                               |
|                                                                                                                                                                 | 0 . 2448 ± 0 . 0039                                                                                                                                   | 0 . 2404 ± 0 . 0025                                                                                               |
| TabM ♠                                                                                                                                                          | 0 . 2469 ± 0 . 0035                                                                                                                                   | 0 . 2440 ± 0 . 0026                                                                                               |
| TabM                                                                                                                                                            | 0 . 2439 ± 0 . 0021                                                                                                                                   | 0 . 2428 ± 0 . 0006                                                                                               |
| TabM[G]                                                                                                                                                         | 0 . 2436 ± 0 . 0027                                                                                                                                   | -                                                                                                                 |
| TabM mini †                                                                                                                                                     | 0 . 2433 ± 0 . 0017                                                                                                                                   | 0 . 2422 ± 0 . 0004                                                                                               |
| TabM mini                                                                                                                                                       | 0 . 2334 0 . 0018                                                                                                                                     | 0 . 2324 0 . 0009                                                                                                 |

±

## maps-routing ↓

ecom-offers ↑

| Method                                                                                                                                                                                             | Single model                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Ensemble                                                                                                                                                                                                                                                                                                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MLP TabPFN ResNet DCN2 SNN Trompt AutoInt MLP - Mixer Excel ∗ SAINT FT - T T2G MLP ‡- lite MLP ‡ MLP † XGBoost LightGBM CatBoost TabR TabR ‡ MNCA MNCA ‡ TabM ♠ TabM TabM[G] TabM mini TabM † mini | 0 . 1625 ± 0 . 0001 - - 0 . 1656 ± 0 . 0004 0 . 1634 ± 0 . 0002 0 . 1624 ± nan - - 0 . 1628 ± 0 . 0001 0 . 1634 ± nan 0 . 1625 ± 0 . 0003 0 . 1616 ± 0 . 0001 0 . 1618 ± 0 . 0002 0 . 1618 ± 0 . 0002 0 . 1620 ± 0 . 0002 0 . 1616 ± 0 . 0001 0 . 1618 ± 0 . 0000 0 . 1619 ± 0 . 0001 0 . 1639 ± 0 . 0003 0 . 1622 ± 0 . 0002 0 . 1625 ± 0 . 0001 0 . 1627 ± 0 . 0002 0 . 1612 ± 0 . 0001 0 . 1612 ± 0 . 0001 0 . 1611 ± 0 . 0001 0 . 1612 ± 0 . 0001 0 . 1610 ± 0 . 0001 | 0 . 1621 ± 0 . 0000 - - 0 . 1636 ± 0 . 0001 0 . 1625 ± 0 . 0000 - - - 0 . 1621 ± nan - 0 . 1619 ± 0 . 0001 0 . 1608 ± nan 0 . 1613 ± 0 . 0000 0 . 1613 ± 0 . 0001 0 . 1614 ± 0 . 0000 0 . 1614 ± 0 . 0000 0 . 1616 ± 0 . 0000 0 . 1615 ± 0 . 0000 0 . 1622 ± 0 . 0002 - 0 . 1621 ± 0 . 0001 0 . 1623 ± 0 . 0001 0 . 1609 ± 0 . 0000 0 . 1610 ± 0 . 0001 - 0 . 1610 ± 0 . 0000 0 . 1609 ± 0 . 0000 |

±

| Method                                                                                                                                              |                                                                                                                                                                                                                                                                                                                                                                     | Ensemble                                                                                                                                                                                                                                                                                    |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MLP TabPFN ResNet DCN2 SNN Trompt AutoInt MLP - Mixer Excel ∗ SAINT FT - T T2G MLP ‡- lite MLP ‡ MLP † XGBoost LightGBM CatBoost TabR TabR ‡ MNCA ‡ | Single model 0 . 5989 ± 0 . 0017 - - 0 . 5996 ± 0 . 0043 0 . 5912 ± 0 . 0056 0 . 5803 ± nan - - 0 . 5759 ± 0 . 0066 0 . 5812 ± 0 . 0098 0 . 5775 ± 0 . 0063 0 . 5791 ± 0 . 0056 0 . 5800 ± 0 . 0029 0 . 5846 ± 0 . 0048 0 . 5949 ± 0 . 0013 0 . 5763 ± 0 . 0072 0 . 5758 ± 0 . 0006 0 . 5596 ± 0 . 0068 0 . 5943 ± 0 . 0019 0 . 5762 ± 0 . 0052 0 . 5765 ± 0 . 0087 | 0 . 5995 ± 0 . 0011 - - 0 . 6039 ± 0 . 0028 0 . 5961 ± 0 . 0033 - - - 0 . 5759 ± nan - 0 . 5817 ± 0 . 0021 0 . 5824 ± nan 0 . 5819 ± 0 . 0011 0 . 5872 ± 0 . 0018 0 . 5953 ± 0 . 0006 0 . 5917 ± 0 . 0035 0 . 5758 ± 0 . 0003 0 . 5067 ± 0 . 0011 0 . 5977 ± 0 . 0009 - 0 . 5820 ± 0 . 0047 |

±

## homesite-insurance ↑

| Method                                                                                                                                                                                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                                                                                            |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MLP TabPFN ResNet DCN2 SNN Trompt AutoInt MLP - Mixer Excel ∗ SAINT FT - T T2G MLP ‡- lite MLP ‡ MLP † XGBoost LightGBM CatBoost TabR TabR ‡ MNCA MNCA ‡ TabM ♠ TabM TabM[G] TabM mini TabM † mini | Single model 0 . 9506 ± 0 . 0005 - - 0 . 9398 ± 0 . 0053 0 . 9473 ± 0 . 0013 0 . 9588 ± nan - - 0 . 9622 ± 0 . 0004 0 . 9613 ± nan 0 . 9622 ± 0 . 0006 0 . 9624 ± 0 . 0006 0 . 9609 ± 0 . 0009 0 . 9617 ± 0 . 0004 0 . 9582 ± 0 . 0014 0 . 9601 ± 0 . 0002 0 . 9603 ± 0 . 0002 0 . 9606 ± 0 . 0003 0 . 9487 ± 0 . 0014 0 . 9556 ± 0 . 0021 0 . 9514 ± 0 . 0038 0 . 9620 ± 0 . 0006 0 . 9641 ± 0 . 0004 0 . 9640 ± 0 . 0002 0 . 9641 ± 0 . 0003 0 . 9643 ± 0 . 0003 0 . 9631 ± 0 . 0003 | Ensemble 0 . 9514 ± 0 . 0001 - - 0 . 9432 ± 0 . 0018 0 . 9484 ± 0 . 0007 - - - 0 . 9635 ± nan - 0 . 9633 ± 0 . 0001 0 . 9637 ± nan 0 . 9626 ± 0 . 0003 0 . 9630 ± 0 . 0002 0 . 9599 ± 0 . 0002 0 . 9602 ± 0 . 0000 0 . 9604 ± 0 . 0001 0 . 9609 ± 0 . 0001 0 . 9505 ± 0 . 0001 - 0 . 9522 ± 0 . 0027 0 . 9635 ± 0 . 0002 0 . 9644 ± 0 . 0003 0 . 9642 ± 0 . 0001 - 0 . 9645 ± 0 . 0001 0 . 9634 ± 0 . 0001 |

cooking-time ↓

±

| Method                                                                                                                                                                                   | Single model                                                                                                                                                                                                                                                                                                                                                                                                                                          | Ensemble                                                                                                                                                                                                                                                                                                                                                                      |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MLP TabPFN ResNet DCN2 SNN Trompt AutoInt MLP - Mixer Excel ∗ SAINT FT - T T2G MLP ‡- lite MLP ‡ MLP † XGBoost LightGBM CatBoost TabR TabR ‡ MNCA MNCA ‡ TabM ♠ TabM TabM[G] TabM mini † | 0 . 4828 ± 0 . 0002 - - 0 . 4834 ± 0 . 0003 0 . 4835 ± 0 . 0006 0 . 4809 ± nan - - 0 . 4821 ± 0 . 0005 0 . 4840 ± nan 0 . 4820 ± 0 . 0008 0 . 4809 ± 0 . 0008 0 . 4811 ± 0 . 0004 0 . 4809 ± 0 . 0006 0 . 4812 ± 0 . 0004 0 . 4823 ± 0 . 0001 0 . 4826 ± 0 . 0001 0 . 4823 ± 0 . 0001 0 . 4828 ± 0 . 0008 0 . 4818 ± 0 . 0006 0 . 4825 ± 0 . 0004 0 . 4818 ± 0 . 0005 0 . 4803 ± 0 . 0006 0 . 4804 ± 0 . 0002 0 . 4800 ± 0 . 0002 0 . 4803 ± 0 . 0001 | 0 . 4822 ± 0 . 0000 - - 0 . 4822 ± 0 . 0001 0 . 4818 ± 0 . 0002 - - - 0 . 4808 ± nan - 0 . 4813 ± 0 . 0005 0 . 4797 ± nan 0 . 4805 ± 0 . 0001 0 . 4804 ± 0 . 0003 0 . 4807 ± 0 . 0002 0 . 4821 ± 0 . 0000 0 . 4825 ± 0 . 0001 0 . 4820 ± 0 . 0001 0 . 4814 ± 0 . 0004 - 0 . 4819 ± 0 . 0003 0 . 4809 ± 0 . 0003 0 . 4797 ± 0 . 0003 0 . 4802 ± 0 . 0000 - 0 . 4801 ± 0 . 0001 |

±

## delivery-eta ↓

| Method                                                                                                                                                                                             | Single model                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Ensemble                                                                                                                                                                                                                                                                                                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MLP TabPFN ResNet DCN2 SNN Trompt AutoInt MLP - Mixer Excel ∗ SAINT FT - T T2G MLP ‡- lite MLP ‡ MLP † XGBoost LightGBM CatBoost TabR TabR ‡ MNCA MNCA ‡ TabM ♠ TabM TabM[G] TabM mini TabM † mini | 0 . 5493 ± 0 . 0007 - - 0 . 5516 ± 0 . 0014 0 . 5495 ± 0 . 0008 0 . 5519 ± nan - - 0 . 5552 ± 0 . 0030 0 . 5528 ± nan 0 . 5542 ± 0 . 0026 0 . 5527 ± 0 . 0016 0 . 5521 ± 0 . 0014 0 . 5535 ± 0 . 0019 0 . 5521 ± 0 . 0019 0 . 5468 ± 0 . 0002 0 . 5468 ± 0 . 0001 0 . 5465 ± 0 . 0001 0 . 5514 ± 0 . 0024 0 . 5520 ± 0 . 0015 0 . 5498 ± 0 . 0007 0 . 5507 ± 0 . 0013 0 . 5510 ± 0 . 0015 0 . 5494 ± 0 . 0004 0 . 5509 ± 0 . 0003 0 . 5497 ± 0 . 0007 0 . 5510 ± 0 . 0019 | 0 . 5478 ± 0 . 0006 - - 0 . 5495 ± 0 . 0004 0 . 5479 ± 0 . 0001 - - - 0 . 5524 ± nan - 0 . 5523 ± 0 . 0018 0 . 5512 ± nan 0 . 5512 ± 0 . 0005 0 . 5526 ± 0 . 0009 0 . 5511 ± 0 . 0007 0 . 5463 ± 0 . 0001 0 . 5465 ± 0 . 0000 0 . 5461 ± 0 . 0000 0 . 5480 ± 0 . 0005 - 0 . 5488 ± 0 . 0002 0 . 5494 ± 0 . 0006 0 . 5504 ± 0 . 0004 0 . 5492 ± 0 . 0001 - 0 . 5495 ± 0 . 0003 0 . 5502 ± 0 . 0000 |

## homecredit-default ↑

±

| Method                                                                                                                                            | Single model                                                                                                                                                                                                                                                                                                                                 | Ensemble                                                                                                                                                                                                                                                                  |
|---------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MLP TabPFN ResNet DCN2 SNN Trompt AutoInt MLP - Mixer Excel ∗ SAINT FT - T T2G MLP ‡- lite MLP ‡ MLP † XGBoost LightGBM CatBoost TabR TabR ‡ MNCA | 0 . 8538 ± 0 . 0014 - - 0 . 8471 ± 0 . 0019 0 . 8541 ± 0 . 0016 0 . 8355 ± nan - - 0 . 8513 ± 0 . 0024 0 . 8377 ± nan 0 . 8571 ± 0 . 0023 0 . 8597 ± 0 . 0007 0 . 8598 ± 0 . 0009 0 . 8572 ± 0 . 0011 0 . 8568 ± 0 . 0039 0 . 8670 ± 0 . 0005 0 . 8664 ± 0 . 0004 0 . 8627 ± nan 0 . 8501 ± 0 . 0027 0 . 8547 ± 0 . 0021 0 . 8531 ± 0 . 0018 | 0 . 8566 ± 0 . 0005 - - 0 . 8549 ± 0 . 0002 0 . 8569 ± 0 . 0010 - - - 0 . 8564 ± nan - 0 . 8611 ± 0 . 0013 0 . 8629 ± nan 0 . 8607 ± 0 . 0003 0 . 8590 ± 0 . 0003 0 . 8614 ± 0 . 0014 0 . 8674 ± 0 . 0001 0 . 8667 ± 0 . 0000 - 0 . 8548 ± 0 . 0003 - 0 . 8569 ± 0 . 0004 |

±

## weather ↓

| Method                                                                                                                                                                                             | Single model                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Ensemble                                                                                                                                                                                                                                                                                                                                                                        |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MLP TabPFN ResNet DCN2 SNN Trompt AutoInt MLP - Mixer Excel ∗ SAINT FT - T T2G MLP ‡- lite MLP ‡ MLP † XGBoost LightGBM CatBoost TabR TabR ‡ MNCA MNCA ‡ TabM ♠ TabM TabM[G] TabM mini TabM † mini | 1 . 5378 ± 0 . 0054 - - 1 . 5606 ± 0 . 0057 1 . 5280 ± 0 . 0085 1 . 5187 ± nan - - 1 . 5131 ± 0 . 0022 1 . 5097 ± 0 . 0045 1 . 5104 ± 0 . 0097 1 . 4849 ± 0 . 0087 1 . 5170 ± 0 . 0040 1 . 5139 ± 0 . 0031 1 . 5162 ± 0 . 0020 1 . 4671 ± 0 . 0006 1 . 4625 ± 0 . 0008 1 . 4688 ± 0 . 0019 1 . 4666 ± 0 . 0039 1 . 4458 ± 0 . 0018 1 . 5062 ± 0 . 0054 1 . 5008 ± 0 . 0034 1 . 4786 ± 0 . 0039 1 . 4722 ± 0 . 0024 1 . 4728 ± 0 . 0022 1 . 4716 ± 0 . 0016 1 . 4651 ± 0 . 0020 | 1 . 5111 ± 0 . 0029 - - 1 . 5292 ± 0 . 0028 1 . 5013 ± 0 . 0034 - - - 1 . 4707 ± nan - 1 . 4719 ± 0 . 0040 1 . 4513 ± nan 1 . 4953 ± 0 . 0023 1 . 4978 ± 0 . 0020 1 . 5066 ± 0 . 0008 1 . 4629 ± 0 . 0002 1 . 4581 ± 0 . 0003 - 1 . 4547 ± 0 . 0008 - 1 . 4822 ± 0 . 0013 1 . 4782 ± 0 . 0011 1 . 4715 ± 0 . 0020 1 . 4675 ± 0 . 0009 - 1 . 4669 ± 0 . 0010 1 . 4581 ± 0 . 0016 |