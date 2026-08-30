# Hybrid ResNet-1D–BiGRU with Multi-Head Attention for Cyberattack Detection in Industrial IoT Environments

Afrah Gueriani *LSEA Lab., Faculty of Technology University of MEDEA* Medea 26000, Algeria gueriani.afrah@univ-medea.dz

Hamza Kheddar *LSEA Lab., Faculty of Technology University of MEDEA* Medea 26000, Algeria kheddar.hamza@univ-medea.dz

Ahmed Cherif Mazari *LSEA Lab, Faculty of Science University of MEDEA* Medea 26000, Algeria mazari.ahmedcherif@univ-medea.dz

*Abstract*—This study introduces a hybrid deep learning model for intrusion detection in Industrial IoT (IIoT) systems, combining ResNet-1D, BiGRU, and Multi-Head Attention (MHA) for effective spatial-temporal feature extraction and attention-based feature weighting. To address class imbalance, SMOTE was applied during training on the Edge-IIoTset dataset. The model achieved 98.71% accuracy, a loss of 0.0417%, and low inference latency (0.0001 sec/instance), demonstrating strong real-time capability. To assess generalizability, the model was also tested on the CICIoV2024 dataset, where it reached 99.99% accuracy and F1-score, with a loss of 0.0028, 0% FPR, and 0.00014 sec/instance inference time. Across all metrics and datasets, the proposed model outperformed existing methods, confirming its robustness and effectiveness for real-time IoT intrusion detection.

*Index Terms*—Intrusion detection system, cyber-attacks ResNet-1D, BiGRU, MHA, Edge-IIoTset

### I. INTRODUCTION

The rapid advancement and extensive deployment of IoT devices have transformed modern life, enhancing convenience and fostering the development of interconnected systems. However, this technological progress also introduces substantial challenges, particularly the increasing vulnerability of IoT devices to cyber threats [\[1\]](#page-4-0). Additionally, the relentless refinement of hacking techniques has given rise to novel and highly sophisticated cyber intrusions, creating complex and unforeseen security challenges [\[2\]](#page-4-1). The IIoT, which extends IoT to industrial applications, faces even greater risks due to its critical role in sectors like manufacturing, healthcare, and energy. This necessitated the development of advanced and robust security measures, including Network Intrusion Detection Systems (NIDSs) (discussed in [\[3\]](#page-4-2)), to identify and alert against attacks and malware at an early stage. Artificial Intelligence (AI), which refers to machines emulating intelligent human behavior [\[4\]](#page-4-3), has increasingly emerged as a pivotal component in the domain of network security and intrusion detection [\[5\]](#page-5-0). This study introduces an innovative hybrid framework that integrates ResNet-1D, BiGRU, and Multihead-Attention mechanisms which is discussed in [\[6\]](#page-5-1) to enhance intrusion detection. Additionally, the use of the SMOTE technique addresses data imbalance in EdgeIIoTset dataset, ensuring more accurate classification. The proposed ResNet-1D-BiGRU-MHA capture spatial and temporal dependencies in IIoT network traffic. The rationale behind selecting this combination is grounded in their complementary strengths in feature extraction, sequence modeling, and attention-based weighting. ResNet-1D is employed as the initial feature extraction component due to its ability to efficiently capture hierarchical representations in sequential network data. ResNet-1D, in contrast to traditional CNNs, uses residual connections to address the vanishing gradient issue, enabling deeper networks to learn without seeing a drop in performance [\[7\]](#page-5-2). Addionaly, BiGRU for temporal dependency modeling is incorporated to model long-term dependencies by reducing computational complexity and allowing the model to learn attack patterns that may depend on both past and future network behavior [\[8\]](#page-5-3). However, MHA is integrated to dynamically assign different levels of importance to various features in the time series data. Even while IDS research has advanced significantly, there are still a number of issues that limit the usefulness of current models, especially in IIoT systems. Security teams are overloaded with false alarms due to the high false positive rates (FPR) of many IDS models. Conversely, false negatives pose a critical risk by allowing sophisticated cyberattacks to go undetected. In realworld cybersecurity datasets, attack samples are often much rarer than normal traffic, leading to class imbalance issues. Many existing models fail to effectively learn from minorityclass attacks, causing reduced detection accuracy for rare but critical threats like in [\[9,](#page-5-4) [10\]](#page-5-5).

This paper's remaining sections are arranged as follows: The preliminary findings, which include a review of the pertinent background and associated studies, are presented in Section [II.](#page-1-0) The suggested method and the dataset preprocessing procedures are explained in Section [III.](#page-1-1) Section [IV](#page-2-0) reports on extensive tests conducted to evaluate the model's performance. A summary of the paper's main conclusions and some avenues for further research are provided in Section [V.](#page-4-4)

#### II. LITERATURE REVIEW

<span id="page-1-0"></span>DL approaches have proven highly effective in addressing cybersecurity threats, outperforming traditional centralized machine learning models, particularly in detecting advanced attacks. To further improve performance, researchers are increasingly adopting hybrid and concatenated DL architectures. Z. Xia et al. [\[11\]](#page-5-6) propose a PSO-GA-optimized ResNet-BiGRU-based intrusion detection method aimed at enhancing network security. To further improve accuracy, genetic algorithm (GA) and particle swarm optimization (PSO) are employed for hyperparameter tuning. Experiments were demonstrate superior performance on three different dataset namely KDD99, UNSW-NB15, and CIC-IDS 2017 compared to existing methods. In [\[12\]](#page-5-7), D. Javeed et al. introduce an explainable and resilient IDS specifically designed for Industry 5.0 environments. The proposed model integrates BiLSTM and BiGRU architectures to enhance detection accuracy. Using the CICDDoS2019 dataset, the system effectively detects and eliminates cyber-threats in interconnected industrial systems. In [\[13\]](#page-5-8), Y. Xiang et al. propose a hybrid DL model combining ResNet and biGRU for effective intrusion detection in IoT environments. Numerous experiments show that the model performs better than current techniques, achieving better detection rates and robustness on three benchmark IoT datasets namely: NBaIoT, PreIoT, and UNSW-NB15. In [\[14\]](#page-5-9), the authors investigated the feasibility of developing a DNN-GRU model enhanced with Multi-Head Attention. The experimental results demonstrated the superiority of the proposed model, achieving accuracy rates of 98.22% and 99.78% on medical and industrial datasets, respectively. In their subsequent works [\[15,](#page-5-10) [16\]](#page-5-11), the same authors employed SHAP with multiple GRU-BiLSTM-MHA-based IDS combinations to analyze and rank feature importance, with the aim of reducing computational cost. Evaluated on the same datasets before and after the application of balancing techniques, the proposed architecture achieved high accuracy across domains, demonstrating strong generalization capability and adaptability under both settings. A Vision Transformer-BiLSTM architecture was also investigated in [\[17\]](#page-5-12) for the development of an advanced IDS method. The experimental results demonstrated that the proposed ViT-BiLSTM model outperformed many existing approaches across multiple evaluation metrics.

#### III. PROPOSED METHODOLOGY

<span id="page-1-1"></span>This section presents a background of attention mechanism, then discuss the suggested model, a ResNet-1D-BiGRU-MHA hybrid, will take up this section. This architecture is created to recognize and classify benign and dangerous traffic in a dataset of novel environments.

*- Attention Mechanism:* The self-attention mechanism is a powerful and highly efficient approach commonly used in modern deep learning architectures, especially in models that process sequential data such as text, audio, or time series. Three vectors; the query, key, and value vectors; represent each component of the input sequence in this technique. Throughout the training process, these vectors are discovered and improved. The core idea is to determine the degree of attention an element (represented by the query) should allocate to another element (represented by the key) by computing a compatibility score, which guides the weighted aggregation of the corresponding value vectors. In order to ensure that the scores total up to one, this score is usually calculated by taking the dot product of the query and key vectors, then adding a scaling factor and performing a softmax operation [\[18\]](#page-5-13).

See Equation 1 in in [\[19\]](#page-5-14) for a thorough description of the attention mechanism:

Attention(Q, K, V) = 
$$softmax\left(\frac{Q \times K^T}{\sqrt{d_q}}\right) \times V$$
 (1)

Where:

• *Q (Query)* represents the transformed input used by the model to compute attention scores. As indicated by Equation [2,](#page-1-2) these vectors are calculated by multiplying the weight matrix W<sup>Q</sup> by the input X and correspond to various segments of the input sequence:

<span id="page-1-2"></span>
$$Q = X \times W_Q \tag{2}$$

• *K (Key)* refers to the set of vectors against which the query is compared to determine relevance. These vectors correspond to various input sequence segments and are computed by multiplying the input X with the weight matrix WK, according to [3:](#page-1-3)

<span id="page-1-3"></span>
$$K = X \times W_K \tag{3}$$

• *V (Values)* are the values associated with each key. These vectors hold the actual information the model will utilize after computing the attention scores, as given by the following equation:

$$V = X \times W_V \tag{4}$$

• *d q* is the dimensionality of the query vector Q.

#### *A. Data Preparation*

Effective data preparation enhances the reliability and accuracy of the proposed models. The following points detail the key steps for data transformation:

- *Data preprocessing:* The initial step for both datasets involved transforming the feature sets; comprising 60 features and 15 distinct attack types in the Edge-IIoT dataset, and 9 features with 6 attack types in the second dataset; into a two-dimensional format to align with the input requirements of the proposed model.
- *Addressing class imbalance with SMOTE:* a popular oversampling technique that creates synthetic instances for the minority class using interpolation from the available samples, was utilized in the study to correct class imbalance [\[20\]](#page-5-15).
- *Data splitting:* Following preprocessing, a total of 60 features from the Edge-IIoT dataset and 9 features from

the CICIoV2024 dataset were retained for model training. Each dataset was subsequently partitioned into training and testing subsets using an 80/20 split, respectively. During the training phase, the Adam optimization technique was used to facilitate effective convergence and raise the model's overall efficacy.

*- Data Numerization:* To facilitate the processing of nonnumeric categorical data, a numerization approach was employed. Specifically, categorical features were converted into numerical representations utilizing the LabelEncoder in conjunction with the fit transform method.

#### *B. Model architecture*

The proposed model combines ResNet-based feature extraction, BiGRU for sequence modeling, and MHA (Figure [1\)](#page-2-1) to enhance intrusion detection in IIoT environments. This hybrid approach captures spatial and temporal dependencies while improving classification accuracy. The proposed architecture begins with the *Input Layer:* The model accepts inputs of shape (60, 1), where 1 denotes a single channel per feature and 60 denotes the number of features. *ResNet Block:* At the beginning of the model, a residual connection with two convolutional layers (Conv1D) is employed for feature extraction. The first Conv1D layer uses 64 filters with a kernel size of 3, followed by BatchNormalization for stabilization. Another Conv1D layer with the same configuration is applied, and a residual connection adds a Conv1D(1x1) layer to match dimensions. *BiGRU* layer with 64 units is applied to capture sequential patterns in the data. The BiGRU processes the output from the convolutional layers, leveraging both forward and backward information from the sequence. *LayerNormalization* is included to stabilize the output from the GRU layer. A *MHA* mechanism with num\_heads=4 and key\_dim=64 is subsequently applied to the output of the BiGRU layer. After that, a *Dropout* layer is used at a rate of 0.5 to randomly deactivate neurons during training to reduce overfitting. The attention mechanism's output is compressed into a one-dimensional vector before being sent to the fully connected layers for categorization. For additional processing and abstraction, two *Dense* layers are employed, each having "ReLU" activation with "64 units" and "32 units", respectively. The six different classes in the multiclass classification task are represented by the "6 units" with a "softmax activation function" in the final output layer.

# <span id="page-2-0"></span>IV. EXPERIMENTATION, OUTCOMES AND DISCUSSION

#### *A. Dataset description*

- Edge-IIoTset dataset[1](#page-2-2) : Emerged as a widely recognized benchmark within the research community for evaluating AI-based IDSs, particularly in real-time applications. This dataset encompasses IoT and IIoT traffic data gathered from a real-world testbed featuring seven interconnected layers and ten smart devices and sensors [\[21\]](#page-5-16). It includes 15 traffic types, grouped into six distinct categories. Initially

<span id="page-2-1"></span>![](_page_2_Figure_7.jpeg)

Fig. 1: The proposed framework architecture.

comprising 1,176 features, the dataset was refined to 61 relevant features focusing on IoT devices.

- CICIoV2024[2](#page-2-3) : Introduced for intrusion detection in the context of the Internet of Vehicles (IoV). It contains five range of modern attack types in addition to normal traffic. The dataset was captured in a realistic testbed environment simulating in-vehicle and vehicular communication scenarios. The dataset comprises three distinct representations; binary, decimal, and hexadecimal; each characterized by a different set of features [\[22\]](#page-5-17).

# *B. Performance metrics*

To evaluate how well the suggested model detects different kinds of attacks, standard assessment metrics. The four basic classification outcomes; true positive (TP), true negative (TN), false positive (FP), and false negative (FN); are the source of these measures, as described in [\[23](#page-5-18)? – [25\]](#page-5-19). In this case, FP and FN stand for the misclassified legitimate and attack occurrences, respectively, whereas TP and TN indicate the number of correctly classified attack and

<span id="page-2-2"></span><sup>1</sup>[https://www.kaggle.com/datasets/cnrieiit/mqttset]( https://www.kaggle.com/datasets/cnrieiit/mqttset)

<span id="page-2-3"></span><sup>2</sup>[https://www.unb.ca/cic/datasets/iov-dataset-2024.html]( https://www.unb.ca/cic/datasets/iov-dataset-2024.html)

valid cases, respectively [\[26\]](#page-5-20). The corresponding equations for each metric are presented below:

$$\begin{split} Acc &= \frac{TP_A + TN_N}{TP_A + FP_N + TN_N + FN_A}, Rc = \frac{TP_A}{TP_A + FN_A} \end{split}$$
 
$$Pr &= \frac{TP_A}{TP_A + FP_N}, \ F1 - Score = 2 \times \frac{Pr \times Rc}{Pr + Rc}$$
 
$$FPR &= \frac{FP_N}{FP_N + TN_N} \end{split}$$

Where the indices A and N refer to abnormal and normal samples, respectively.

#### *C. Experiments and Results*

The experimental findings of assessing the suggested model using a variety of performance measures on the Edge-IIoTset dataset are shown in this section. The training was conducted in the Kaggle environment utilizing a dual GPU T4 setup with 15 GiB of memory, using the Adam optimizer for fifteen epochs.

• *Accuracy and loss graph:* Figure [2](#page-3-0) Figure 2 (a) and (b) display the training/validation accuracy and loss of the proposed model. The validation accuracy reaches 98.71% and closely follows the training accuracy, suggesting good generalization without overfitting. Both losses stabilize early and remain low, with the validation loss settling at 0.0417, indicating effective model optimization.

<span id="page-3-0"></span>![](_page_3_Figure_6.jpeg)

Fig. 2: Accuracy and loss of the proposed ResNet-1D-BiGRU-MHA model. (a): Accuracy graph, (b): Loss graph

• *Classification Report:* Table [I](#page-3-1) shows the proposed model's strong performance in multiclass attack detection. It achieves perfect accuracy for normal traffic, information gathering, MITM, and malware attacks, and performs well on DDoS and injection attacks with minimal misclassifications. Overall, the ResNet-1D-BiGRU-MHA architecture

<span id="page-3-1"></span>TABLE I: Multiclass classification report of the proposed ResNet-1D-BiGRU-MHA approach.

|                     | Precision | Recall | F1   | Support |
|---------------------|-----------|--------|------|---------|
| Benign traffic (0)  | 100%      | 100%   | 100% | 9860    |
| DDoS (1)            | 99%       | 97%    | 98%  | 10016   |
| Info. gathering (2) | 100%      | 100%   | 100% | 9878    |
| MITM (3)            | 100%      | 100%   | 100% | 9815    |
| Injection (4)       | 97%       | 99%    | 98%  | 9779    |
| Malware (5)         | 100%      | 100%   | 100% | 9928    |

demonstrates effective classification capability and is wellfor real-time cybersecurity threat detection.

• *Confusion matrix:* Figure [3](#page-3-2) shows the confusion matrix for the ResNet-1D-BiGRU-MHA model, demonstrating strong classification performance. Most attack types are correctly identified, with high accuracy along the diagonal and minimal misclassifications. Notably, only 4% of "malware" attacks are misclassified as "DDoS", and 2% of "DDoS" attacks as "malware", indicating effective differentiation between attack types.

<span id="page-3-2"></span>![](_page_3_Figure_13.jpeg)

Fig. 3: Confusion matrix of the proposed ResNet-1D-BiGRU-MHA model

- *Inference and training time:* In real-time IDS, the primary emphasis is placed on inference time rather than training duration. In this study, the ResNet-1D-BiGRU-MHA model achieves an inference time of 0.0001 seconds per instance, while the ResNet-1D-BiGRU-MHA model requires 58.07 seconds per epoch for training. These results highlight the exceptionally fast performance of the proposed models, particularly during inference.
- *FPR:* A value of 0.002% in multiclass attack detection indicates an exceptionally low rate of false alarms, proving the accuracy of the model in differentiating between harmful and benign traffic.
- *Receiver operating characteristics (ROC):* Figure [4](#page-4-5) presents the ROC curves for the proposed multiclass attack detection model. The ROC curves for all classes (0 to 5) are positioned at the top-left corner of the plot, indicating ideal classification performance. An AUC value

of 1.00 across all classes further confirms that the model achieves perfect accuracy in distinguishing true positives and true negatives for each attack category.

<span id="page-4-5"></span>![](_page_4_Figure_1.jpeg)

Fig. 4: ROC curves for multiclass classification of the proposed ResNet-1D- BiGRU-MHA model.

• *Performance Evaluation Against Existing Methods:* Table [II](#page-5-21) presents a comparative analysis of the proposed ResNet-1D-BiGRU-MHA model against recent state-ofthe-art approaches on the Edge-IIoTSet and CICIoV2024 datasets. The proposed model consistently outperforms existing methods across all major evaluation metrics.

On Edge-IIoTSet, it achieves an accuracy of 98.71%, surpassing models like BiGRU-LSTM [\[27\]](#page-5-22). Although LSTM-CNN-Att [? ] slightly exceeds this with 99.04% accuracy, it was tested on a single dataset and lacks inference time reporting, limiting its practical evaluation. In contrast, the proposed model demonstrates high performance and low inference latency (0.0001 sec/instance) across both datasets, confirming its generalizability and real-time suitability.

On CICIoV2024, the model achieves 99.99% accuracy, with near-perfect precision, recall, F1-score, and 0.0000% FPR, significantly outperforming models such as CNN-LSTM-ViT [\[28\]](#page-5-23) and XGB [\[29\]](#page-5-24). Its inference time on this dataset is also the lowest (0.00014 sec/instance) compared to 0.0213 sec for CNN-LSTM-ViT, highlighting its computational efficiency.

- *Ablation study:* To evaluate the contribution of individual architectural components to the overall performance of the proposed model, an ablation study was conducted. Various model configurations were analyzed, with the results summarized in Table III [III.](#page-5-25) The analyzed configurations include: ResNet-1D, BiGRU-MHA, ResNet-1D-BiGRU, and ResNet-1D-BiGRU-MHA, with varying numbers of attention heads, dropout rates, and dense layers.
  - The baseline model (ResNet-1D, #1) showed limited performance (47.95% accuracy), highlighting the absence of temporal feature extraction.
  - Adding BiGRU with MHA (#2) improved accuracy, though the loss remained high, indicating further enhancement was needed.
  - Model #3 combined ResNet-1D and BiGRU, improving temporal learning, but with loss values similar to #2,

- suggesting attention mechanisms could enhance performance.
- The proposed full model (#5), integrating ResNet-1D, BiGRU, and MHA (with four attention heads and 0.5 dropout), achieved the best results; 98.71% accuracy, low FPR, and minimal inference time.
- Varying attention heads showed that reducing heads to two (#4) maintained high performance, while increasing to eight (#6) led to decreased accuracy, suggesting an optimal head count.
- Dropout variations in models #7 (0.3) and #8 (0.7) had minimal effect, indicating robustness to dropout rate changes in this range.
- Model #10, identical to #5 but without SMOTE, showed degraded performance, underlining the importance of class imbalance handling.

This analysis confirms the synergistic effect of combining spatial (ResNet-1D), temporal (BiGRU), and attention mechanisms (MHA), along with balanced data processing, in achieving optimal model performance.

#### V. CONCLUSION

<span id="page-4-4"></span>The proposed model combines SMOTE, MHA, and a hybrid ResNet-1D-BiGRU architecture to effectively detect cyberattacks in IIoT environments. Validated on the Edge-IIoTset dataset, the model achieves over 98% accuracy, high F1-scores, low FPR, and minimal inference times; making it highly appropriate for real-time intrusion detection.

A detailed ablation study confirms the contribution of each architectural component to overall performance. Further evaluation on the CICIoV2024 dataset shows the model maintains high accuracy (over 99%), outperforming previous approaches and demonstrating strong generalization.

While the results are promising, future work should explore broader dataset evaluations and address evolving threats. Potential directions include incorporating transfer learning for adaptability, reinforcement learning for hyperparameter tuning, and Explainable AI (XAI) to enhance interpretability and reliability of the system.

# ACKNOWLEDGMENT

The authors acknowledge that the study was partially funded by the PRFU-A25N01UN260120230001 grant from the Algerian Ministry of Higher Education and Scientific Research.

#### REFERENCES

- <span id="page-4-0"></span>[1] A. Gueriani, H. Kheddar, and A. C. Mazari, "Adaptive cyber-attack detection in iiot using attention-based LSTM-CNN models," in *2024 International Conference on Telecommunications and Intelligent Systems (ICTIS)*. IEEE, 2024, pp. 1–6.
- <span id="page-4-1"></span>[2] M. A. Hossain and M. S. Islam, "Ensuring network security with a robust intrusion detection system using ensemble-based machine learning," *Array*, vol. 19, p. 100306, 2023.
- <span id="page-4-2"></span>[3] H. Kheddar, D. W. Dawoud, A. I. Awad, Y. Himeur, and M. K. Khan, "Reinforcement-learning-based intrusion detection in communication networks: A review," *IEEE Communications Surveys & Tutorials*, 2024.
- <span id="page-4-3"></span>[4] M. Kubat, *Fundamentals of Artificial Intelligence: Problem Solving and Automated Reasoning*. McGraw-Hill Education, 2023.

<span id="page-5-21"></span>TABLE II: Comparing the Best Practices for Multiclass Classification on the Edge-IIoTset Dataset with Performance Metrics for the Suggested ResNet-1D-BiGRU-MHA Model.

| Work      | Model               | Dataset    | Acc (%) | Loss   | Pr (%) | Rc (%) | F1 (%) | FPR (%) | Inf time (Sec/Inst) |  |
|-----------|---------------------|------------|---------|--------|--------|--------|--------|---------|---------------------|--|
| [? ]      | LSTM-CNN-Att        | EdgeIIoT   | 99.04   | 0.0220 | 99.05  | 99.04  | 99.04  | 0.002   | ✗                   |  |
| [22]      | DNN                 | CICIoV2024 | 96      | ✗      | 83     | 76     | 78     | ✗       | ✗                   |  |
| [27]      | BiGRU-LSTM          | EdgeIIoT   | 98.32   | ✗      | 98.78  | 97.22  | ✗      | ✗       | ✗                   |  |
| [28]      | CNN-LSTM-ViT        | CICIoV2024 | 99.78   | ✗      | ✗      | ✗      | 99.65  | 1.2     | 0.0213              |  |
| Presented | ResNet-1D-BiGRU-MHA | EdgeIIoT   | 98.71   | 0.0417 | 98.71  | 98.70  | 98.71  | 0.002   | 0.0001              |  |
|           |                     | CICIoV2024 | 99.99   | 0.0028 | 99.99  | 99.99  | 99.99  | 0.0000  | 0.00014             |  |

TABLE III: Performance of different variants of the proposed models in multiclass classification.

<span id="page-5-25"></span>

| Case number | Model                    | N. of Att heads | Dropout (%) | Accuracy (%) | Loss(%) | FPR(%) | Inf. time |
|-------------|--------------------------|-----------------|-------------|--------------|---------|--------|-----------|
| #1          | ResNet-1D                | ✗               | 0.5         | 47.95        | 2.7668  | 0.0130 | 0.00008   |
| #2          | BiGRU-MHA                | 4               | 0.5         | 98.28        | 0.0560  | 0.0034 | 0.0002    |
| #3          | ResNet-1D-BiGRU          | ✗               | 0.5         | 98.07        | 0.0557  | 0.0039 | 0.0003    |
| #4          | ResNet-1D-BiGRU-MHA      | 2               | 0.5         | 98.27        | 0.0504  | 0.0033 | 0.00015   |
| #5          | ResNet-1D-BiGRU-MHA      | 4               | 0.5         | 98.71        | 0.0294  | 0.0020 | 0.0001    |
| #6          | ResNet-1D-BiGRU-MHA      | 8               | 0.5         | 97.97        | 0.0613  | 0.0038 | 0.00016   |
| #7          | ResNet-1D-BiGRU-MHA      | 4               | 0.3         | 97.95        | 0.0650  | 0.0025 | 0.00017   |
| #8          | ResNet-1D-BiGRU-MHA      | 4               | 0.7         | 98.38        | 0.0524  | 0.0033 | 0.00017   |
| #9          | ResNet-1D-BiGRU-MHA      | 4               | 0.5         | 98.30        | 0.0500  | 0.0033 | 0.00017   |
|             | with only 2 dense layers |                 |             |              |         |        |           |
| #10         | ResNet-1D-BiGRU-MHA      | 4               | 0.5         | 96.97        | 0.0873  | 0.0053 | 0.00015   |
|             | without SMOTE technique  |                 |             |              |         |        |           |

- <span id="page-5-0"></span>[5] B. Sharma, L. Sharma, C. Lal, and S. Roy, "Explainable artificial intelligence for intrusion detection in iot networks: A deep learning based approach," *Expert Systems with Applications*, vol. 238, p. 121751, 2024.
- <span id="page-5-1"></span>[6] H. Kheddar, "Transformers and large language models for efficient intrusion detection systems: A comprehensive survey," *Information Fusion*, vol. 124, p. 103347, 2025.
- <span id="page-5-2"></span>[7] G. Zhao, C. Ren, J. Wang, Y. Huang, and H. Chen, "Iot intrusion detection model based on gated recurrent unit and residual network," *Peer-to-Peer Networking and Applications*, vol. 16, no. 4, pp. 1887– 1899, 2023.
- <span id="page-5-3"></span>[8] A. Drewek-Ossowicka, M. Pietrołaj, and J. Ruminski, "A survey of ´ neural networks usage for intrusion detection systems," *Journal of Ambient Intelligence and Humanized Computing*, vol. 12, no. 1, pp. 497–514, 2021.
- <span id="page-5-4"></span>[9] J. Zhou, W. Fu, H. Song, S. Yu, Q. Xuan, and X. Yang, "Multi-view correlation-aware network traffic detection on flow hypergraph," *arXiv preprint arXiv:2501.08610*, 2025.
- <span id="page-5-5"></span>[10] M. L. Hernandez-Jaimes, A. Martinez-Cruz, K. A. Ram´ırez-Gutierrez, ´ and A. Morales-Reyes, "Network traffic inspection to enhance anomaly detection in the internet of things using attention-driven deep learning," *Integration*, p. 102398, 2025.
- <span id="page-5-6"></span>[11] Z. Xia, S. He, C. Liu, Y. Liu, X. Yang, and H. Bu, "Pso-ga hyperparameter optimized resnet-bigru based intrusion detection method," *IEEE Access*, 2024.
- <span id="page-5-7"></span>[12] D. Javeed, T. Gao, P. Kumar, and A. Jolfaei, "An explainable and resilient intrusion detection system for industry 5.0," *IEEE Transactions on Consumer Electronics*, vol. 70, no. 1, pp. 1342–1350, 2023.
- <span id="page-5-8"></span>[13] Y. Xiang, D. Li, X. Meng, C. Dong, and G. Qin, "Resnest-bigru: An intrusion detection model based on internet of things." *Computers, Materials & Continua*, vol. 79, no. 1, 2024.
- <span id="page-5-9"></span>[14] A. Gueriani, H. Kheddar, and A. C. Mazari, "Cyber threat detection in iiot and iomt using dnn-gru with multi-head attention," in *2025 International Conference on Research in Computing at Feminine (RIF)*. IEEE, 2025, pp. 1–8.
- <span id="page-5-10"></span>[15] ——, "Explainable bilstm-mha-based ids for iot using shap and zeroday attack detection," in *2025 International Conference on Artificial Intelligence and Innovative Applications (AIIA)*. IEEE, 2025, pp. 1–8.
- <span id="page-5-11"></span>[16] A. Gueriani, H. Kheddar, A. C. Mazari, and M. C. Ghanem, "A robust cross-domain ids using bigru-lstm-attention for medical and industrial iot security," *ICT Express*, 2025.
- <span id="page-5-12"></span>[17] A. Gueriani, H. Kheddar, A. C. Mazari, S. Sagiroglu, and O. Ceran, "Se-enhanced vit and bilstm-based intrusion detection for secure iiot and iomt environments," in *2025 18th International Conference on Information Security and Cryptology (ISCTurkiye) ¨* . IEEE, 2025, pp. 1–6.

- <span id="page-5-13"></span>[18] A. Vaswani, "Attention is all you need," *Advances in Neural Information Processing Systems*, 2017.
- <span id="page-5-14"></span>[19] M. A. Qathrady, S. Ullah, M. S. Alshehri, J. Ahmad, S. Almakdi, S. M. Alqhtani, M. A. Khan, and B. Ghaleb, "Sacnn-ids: A selfattention convolutional neural network for intrusion detection in industrial internet of things," *CAAI Transactions on Intelligence Technology*, vol. 9, no. 6, pp. 1398–1411, 2024.
- <span id="page-5-15"></span>[20] H. R. Sayegh, W. Dong, and A. M. Al-madani, "Enhanced intrusion detection with lstm-based model, feature selection, and smote for imbalanced data," *Applied Sciences*, vol. 14, no. 2, p. 479, 2024.
- <span id="page-5-16"></span>[21] M. A. Ferrag, O. Friha, D. Hamouda, L. Maglaras, and H. Janicke, "Edge-iiotset: A new comprehensive realistic cyber security dataset of iot and iiot applications for centralized and federated learning," *IEEE Access*, vol. 10, pp. 40 281–40 306, 2022.
- <span id="page-5-17"></span>[22] E. Carlos Pinto Neto, H. Taslimasa, S. Dadkhah, S. Iqbal, P. Xiong, T. Rahman, and A. Ghorbani, "Ciciov2024: Advancing realistic ids approaches against dos and spoofing attack in iov can bus," *Hamideh and Dadkhah, Sajjad and Iqbal, Shahrear and Xiong, Pulei and Rahman, Taufiq and Ghorbani, Ali, Ciciov2024: Advancing Realistic Ids Approaches Against Dos and Spoofing Attack in Iov Can Bus*, 2024.
- <span id="page-5-18"></span>[23] A. Gueriani, H. Kheddar, and A. C. Mazari, "Deep reinforcement learning for intrusion detection in iot: A survey," in *2023 2nd International Conference on Electronics, Energy and Measurement (IC2EM)*, vol. 1. IEEE, 2023, pp. 1–7.
- [24] H. Kheddar, M. Hemis, Y. Himeur, D. Meg´ıas, and A. Amira, "Deep learning for steganalysis of diverse data types: A review of methods, taxonomy, challenges and future directions," *Neurocomputing*, p. 127528, 2024.
- <span id="page-5-19"></span>[25] H. Kheddar, Y. Himeur, S. Al-Maadeed, A. Amira, and F. Bensaali, "Deep transfer learning for automatic speech recognition: Towards better generalization," *Knowledge-Based Systems*, vol. 277, p. 110851, 2023.
- <span id="page-5-20"></span>[26] C. Dunn, N. Moustafa, and B. Turnbull, "Robustness evaluations of sustainable machine learning models against data poisoning attacks in the internet of things," *Sustainability*, vol. 12, no. 16, p. 6434, 2020.
- <span id="page-5-22"></span>[27] D. Javeed, T. Gao, M. S. Saeed, and P. Kumar, "An intrusion detection system for edge-envisioned smart agriculture in extreme environment," *IEEE Internet of Things Journal*, 2023.
- <span id="page-5-23"></span>[28] N. A. Jailani, R. Kumar, and S. Tyagi, "A hybrid deep learning framework for multi-modal intrusion detection in internet of vehicles," in *2025 3rd International Conference on Sustainable Computing and Data Communication Systems (ICSCDS)*. IEEE, 2025, pp. 900–906.
- <span id="page-5-24"></span>[29] F. C¸ olhak, H. Cos¸kun, T. N. R. Cyrille, T. Hoxa, M. ˙I. Ecevit, and M. N. Aydın, "Accelerating iov intrusion detection: Benchmarking gpu-accelerated vs cpu-based ml libraries," *arXiv preprint arXiv:2504.01905*, 2025.