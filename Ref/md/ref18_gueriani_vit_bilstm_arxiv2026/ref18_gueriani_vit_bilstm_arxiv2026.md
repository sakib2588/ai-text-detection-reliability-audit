# SE-Enhanced ViT and BiLSTM-Based Intrusion Detection for Secure IIoT and IoMT Environments

1 st Afrah Gueriani *LSEA Lab., Faculty of Technology University of MEDEA* Medea 26000, Algeria gueriani.afrah@univ-medea.dz

4 th Seref Sagiroglu *Department of Computer Engineering Gazi University* Ankara, Turkey ss@gazi.edu.tr

2 nd Hamza Kheddar *LSEA Lab., Faculty of Technology University of MEDEA* Medea 26000, Algeria kheddar.hamza@univ-medea.dz

3 rd Ahmed Cherif Mazari *LSEA Lab, Faculty of Science University of MEDEA* Medea 26000, Algeria mazari.ahmedcherif@univ-medea.dz

5 th Onur Ceran *Department of Management Information Systems Gazi University* Ankara, Turkey onur.ceran@gazi.edu.tr

*Abstract*—With the rapid growth of interconnected devices in Industrial and Medical Internet of Things (IIoT and MIoT) ecosystems, ensuring timely and accurate detection of cyber threats has become a critical challenge. This study presents an advanced intrusion detection framework based on a hybrid Squeeze-and-Excitation Attention Vision Transformer-Bidirectional Long Short-Term Memory (SE ViT-BiLSTM) architecture. In this design, the traditional multi-head attention mechanism of the Vision Transformer is replaced with Squeezeand-Excitation attention, and integrated with BiLSTM layers to enhance detection accuracy and computational efficiency. The proposed model was trained and evaluated on two realworld benchmark datasets; EdgeIIoT and CICIoMT2024; both before and after data balancing using the Synthetic Minority Over-sampling Technique (SMOTE) and RandomOverSampler. Experimental results demonstrate that the SE ViT-BiLSTM model outperforms existing approaches across multiple metrics. Before balancing, the model achieved accuracies of 99.11% (FPR: 0.0013%, latency: 0.00032 sec/inst) on EdgeIIoT and 96.10% (FPR: 0.0036%, latency: 0.00053 sec/inst) on CICIoMT2024. After balancing, performance further improved, reaching 99.33% accuracy with 0.00035 sec/inst latency on EdgeIIoT and 98.16% accuracy with 0.00014 sec/inst latency on CICIoMT2024.

*Index Terms*—Squeeze-and-Excitation attention, Vision Transformer, Bidirectional Long Short-Term Memory, Intrusion Detection System, Industrial Internet of Things, Internet of Medical Things, Cybersecurity, Network traffic analysis.

# I. INTRODUCTION

The rapid expansion of the Internet of Things (IoT) has transformed various industries by enabling seamless connectivity, efficient data collection, and automation [1]. This has led to the development of specialized domains such as the Industrial Internet of Things (IIoT) and the Internet of Medical Things (IoMT), which enhance operational efficiency through automated data analysis, predictive maintenance, and informed decision-making [2]–[4]. IoMT systems, in particular, enable real-time patient data collection and transmission to healthcare professionals, improving diagnostic accuracy and reducing human error [5], [6]. However, the heterogeneous and highly connected nature of these networks increases vulnerability to cyber threats, putting data confidentiality, availability, and system integrity at risk [7]. As IoT becomes more integrated into critical infrastructure, the need for robust and intelligent Intrusion Detection Systems (IDS) becomes essential to maintain security and reliability in dynamic IoMT and IIoT environments [8]. Traditional security mechanisms often struggle to detect advanced or evolving cyber threats. Intrusion Detection Systems (IDS) address this challenge by continuously monitoring network activity and analyzing abnormal behavior [9]. Common IDS techniques in IoT include signaturebased, anomaly-based, and behavior-based detection. Anomaly detection, in particular, plays a key role in identifying potential threats by analyzing network traffic for unusual patterns [10], enabling timely security responses [10], [11]. To enhance IDS capabilities, this study proposes a hybrid deep learning (DL) model that combines a squeeze-and-excitation (SE) enhanced Vision Transformer (ViT) with a Bidirectional Long Short-Term Memory (BiLSTM) network. This architecture leverages ViT's global feature modeling, BiLSTM's temporal learning, and SE's adaptive feature recalibration to improve detection accuracy.

# *A. Our Contribution*

The key contributions of our work include:

- Design a novel hybrid IDS architecture that integrates SE blocks with a ViT and BiLSTM, aiming to enhance both spatial feature extraction and temporal dependency modeling.
- Evaluate the proposed model using two benchmark datasets: the Edge-IIoT2024 dataset and a subset of the augmented CIC-IoMT2024 dataset.
- To address class imbalance, SMOTE is applied to the Edge-IIoT2024 dataset, and RandomOverSampler is used for the CIC-IoMT2024 subset. Comparative experiments are performed before and after balancing.

- The study targets a multiclass classification task, aiming to accurately detect and classify various cyber attack types across different IoT environments.
- An in-depth evaluation is conducted using key performance metrics, showing the model's robustness and effectiveness in identifying complex attack patterns compared to state-of-the-art methods.

#### *B. Organization of the paper*

The rest of the paper is structured as follows: Section II presents the problem statement and similar works from the literature. Section III provides the proposed methodology and dataset prepossessing. Section IV discusses the experimentation, results, and discussion. Section V concludes the paper and mentions future work.

#### II. PROBLEM STATEMENT AND LITERATURE REVIEW

The rapid expansion of interconnected IoT devices has created major security challenges due to their diverse and resource-limited nature. These systems are highly susceptible to threats such as malware, unauthorized access, and data breaches. The core issues include the dynamic and heterogeneous structure of IoT environments, constantly changing network topologies, and the need to both prevent attacks and minimize damage from successful intrusions.

In this context, several state-of-the-art works, have addressed these issues by developing advanced solutions and methodologies, such as the study by O. Ceran et al. [12], propose an innovative hybrid intrusion detection framework that integrates Graph Neural Networks (GNNs) with the XG-Boost algorithm to effectively address the evolving security challenges in IoT environments. The experimental evaluation conducted on four real-world datasets namely: CICIoT-2023, CICIDS-2017, UNSW-NB15, and IoMT-2024, demonstrates that the proposed hybrid model consistently outperforms conventional machine learning (ML) approaches across various performance metrics. Considering [13], H. Peng et al. propose a BiLSTM-based IDS tailored for IoT environments. To enhance performance, the authors integrate mutual information for feature selection and apply focal loss to address class imbalance during training. Evaluated on EdgeIIoT benchmark dataset, the proposed IDS demonstrates superior accuracy and robustness compared to traditional methods. Another work by A. Gueriani et al. in [14] propose a robust cross-domain IDS that integrates BiGRU, LSTM, and attention mechanisms to enhance security in IoMT and IIoT environments, and in [15], the same authors employed ResNet-1D-BiGRU with multihead Attention. Evaluated on Edge-IIoTset and CICIoMT2024 datasets before and after balancing techniques, the architecture achieves high accuracy across domains, demonstrating strong generalization and adaptability for both schemes. However, [14] highlights its potential for zero-day attacks in diverse IoT infrastructures and extend the study to explainable artificial intelligence (XAI).

#### III. MATERIALS AND METHOD

### *A. Data preprocessing*

To ensure consistency and optimize the performance of the proposed intrusion detection models, a standardized data preprocessing pipeline was applied to both the Edge-IIoTset and CICIoMT2024 datasets. The following steps were conducted:

- *Label Encoding:* To support multi-class classification tasks, the *Attack type* column was mapped to integer values using a predefined dictionary.
- *Categorical Feature Encoding:* Several protocol-related categorical features were identified and transformed using *LabelEncoder* from the scikit-learn library.
- *Feature Selection and Cleaning:* Non-contributory or redundant columns such as *Attack type*, *Attack label*, and *frame.time* were excluded from the feature set "During the feature selection phase, the Attack type column was removed from the input features, as it represents the target class. After feature selection, the Attack type column was reshaped and encoded separately to serve as the label for model training in the supervised learning process3.
- *-Target Variable Encoding:* The *Attack type* column, used as the classification target was reshaped and label-encoded into a numerical format, preparing it for multi-class classification scenarios.
- *Addressing Class Imbalance:* To mitigate the effects of class imbalance, prevalent in both EdgeIIoT and CICIoMT2024 datasets, the *SMOTE* and *RandomOverSampler* were applied respectively for both datasets.
- *Train-Validation Splitting:* The preprocessed selected sets of the datasets are divided into two segments: 80% of the preprocessed data was allocated to training our proposed model and 20% of the preprocessed data was reserved to validate the performance of the enhanced SE-based ViT-BiLSTM model.

#### *B. Structural model*

To effectively capture both spatial dependencies and sequential patterns in network traffic data, we propose a hybrid DL architecture that integrates a SE-based ViT block with a BiLSTM network (Figure 1).

- Input Layer: The model accepts a dimensional input of shape (60, 1) for EdgeIIoT and (83,1) for CICIoMT2024, where 60 and 83 represent the number of selected features for each dataset and 1 denotes the time dimension.
- Branch 1 (SE-ViT Pathway): This branch is responsible for feature embedding and attention-based feature recalibration. A Dense layer with ReLU activation projects the input into a higher-dimensional embedding space. A custom ViT block processes the embedded features as a sequence of tokens. However, instead of standard selfattention, we incorporate a SE attention mechanism, which: Performs global average pooling to capture aggregated channel information. than Passes the result through two Dense layers with relu and sigmoid activation to generate channel-wise importance weights. and recalibrates the features by rescaling them with these weights.

![](_page_2_Figure_0.jpeg)

Fig. 1. Architectural workflow for Proposed Model.

- The recalibrated features are added back to the original input through a residual connection and normalized using Layer Normalization.
- The final output is flattened to prepare it for fusion with the BiLSTM pathway.
- Branch 2 (BiLSTM Pathway): This branch captures temporal dependencies in the feature sequence. Bidirectional LSTM layer with 32 units is used to process the input sequence in both forward and backward directions. The return sequences=True setting ensures that the model retains the full temporal dynamics across time steps. The output sequence is then flattened to align with the output from the SE-ViT branch.
- Feature Fusion and Output Layer: The flattened outputs from both branches are concatenated to form a comprehensive feature vector that captures both spatial and temporal dependencies. This vector is passed through a Dense output layer with softmax activation to classify the input into one of six output classes.

#### IV. EXPERIMENTATION, RESULTS AND DISCUSSION

#### *A. Dataset description*

- *EdgeIIoT dataset*<sup>1</sup> *:* The Edge-IIoTset dataset, introduced in 2022 by Ferrag et al. [16], serves as a comprehensive and publicly available benchmark for evaluating cyber security mechanisms in real-world IoT and IIoT environments. The dataset features 14 attack types grouped into five major categories.
- *CICIoMT2024 dataset*<sup>2</sup> *:* introduced by the Canadian Institute for Cyber security (CIC) in 2024 [17], represents a robust and realistic benchmark tailored for the evaluation of IDS within IoMT environments. It features 18 distinct attack types, systematically grouped into five major categories.

![](_page_2_Figure_10.jpeg)

Fig. 2. Accuracy and loss of the proposed model. (a): Accuracies before balancing the dataset, (b): Loss before balancing the dataset, (c): Accuracies after balancing the dataset, (d): Loss after balancing the dataset.

#### *B. Performance metrics*

To comprehensively assess the performance of the proposed SE ViT-BiLSTM model-based IDS, a range of standard classification metrics was employed. These metrics are derived from the fundamental confusion matrix components: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN). These metrics are formally defined and described in detail in [18]–[22].

#### *C. Experiments and results*

The experiments used the Edge-IIoTset and CICIoMT2024 datasets, trained on Google Colab with a T4 GPU and 12 GB RAM over 50 epochs using the Adam optimizer and categorical cross-entropy. Table I compares model performance before and after data balancing. Initially, the model achieved strong results on Edge-IIoT (99.11% accuracy, low loss and FPR) but slightly lower performance on CICIoMT2024 (96.10% accuracy, higher loss) due to class imbalance. After balancing,

<sup>1</sup>https://www.kaggle.com/code/mohamedamineferrag/ edge-iiotset-pre-processing

<sup>2</sup>https://www.unb.ca/cic/datasets/iomt-dataset-2024.html

TABLE I COMPREHENSIVE EVALUATION RESULTS OF THE ENHANCED SE VIT-BILSTM MODEL USING EDGEIIOT AND CICIOMT2024 DATASETS.

| B   | Dataset           | Acc   | Loss   | Pr    | Rc    | F1    | FPR    | Inf<br>time |
|-----|-------------------|-------|--------|-------|-------|-------|--------|-------------|
|     |                   | (%)   |        | (%)   | (%)   | (%)   | (%)    | (sec/ins)   |
| No  | EdgeIIoT          | 99.11 | 0.0247 | 99.11 | 99.11 | 99.11 | 0.0013 | 0.00032     |
|     | CICIoMT2024 96.10 |       | 0.1440 | 96.10 | 96.10 | 96.10 | 0.0223 | 0.00053     |
| Yes | EdgeIIoT          | 99.33 | 0.0158 | 99.33 | 99.33 | 99.33 | 0.0013 | 0.00035     |
|     | CICIoMT2024 98.16 |       | 0.0578 | 98.16 | 98.16 | 98.15 | 0.0036 | 0.00014     |

Abbreviation: Balancing (B)

both datasets improved; Edge-IIoT reached 99.33% accuracy with reduced loss and low FPR, while CICIoMT2024 rose to 98.16% accuracy with notable decreases in loss and FPR. Precision, recall, and F1-scores also increased, and inference time remained low, showing that data balancing enhanced both performance and efficiency.

- *Accuracy and loss graph:* The presented results in Figure 2 with four subfigures (a)-(d) illustrate the training and validation performance of the enhanced SE ViT-BiLSTM model in terms of accuracy and loss. Before balancing: Accuracy (a): The model reaches high accuracy quickly. EdgeIIoT achieves 99.11%, while CICIoMT2024 is slightly lower at 96.10%, reflecting class imbalance impact. Loss (b): EdgeIIoT shows low, stable loss (0.0247). CICIoMT2024 shows higher, fluctuating loss (0.1440), indicating less stable learning. After Balancing: Accuracy (c): Accuracy improves and aligns across datasets; 99.33% for EdgeIIoT and 98.16% for CICIoMT2024. Loss (d): Loss values decrease and stabilize (0.0158 for EdgeI-IoT, 0.0578 for CICIoMT2024), showing more consistent and effective learning.
- *Confusion matrix:* Figure 3 (subfigures a-d) presents confusion matrices showing the IDS model's performance on the EdgeIIoT and CICIoMT2024 datasets before and after data balancing.

Before balancing, the model performs nearly perfectly on EdgeIIoT (subfigure a) but shows noticeable misclassifications on CICIoMT2024 (subfigure b), particularly within the MQTT class. After balancing, EdgeIIoT performance remains consistently high (subfigure c), while CICIoMT2024 shows major improvement (subfigure d), with reduced confusion in the MQTT class and near-perfect classification across all categories, including previously problematic ones like Recon and DDoS. These results highlight the strong positive effect of data balancing on multiclass classification performance.

*- ROC curve:* The presented ROC curves in Figure 4, subfigures (a)-(d) offer a detailed visualization of the classification performance. Before Balancing; on EdgeIIoT, perfect classification with AUC of 1.00 for all classes. However, on CICIoMT2024, most classes perform well (AUC = 1.00), but Class 0 and Class 2 show lower AUCs (0.98 and 0.92), indicating weaker detection, especially for MQTT-related attacks. After Balancing; EdgeIIoT, maintains perfect AUC (1.00) for all classes and CICIoMT2024, Significant improvement; all classes, including the previously weak Class 2, reach AUC =

![](_page_3_Figure_8.jpeg)

Fig. 3. Confusion matrix of the proposed model. (a): Before balancing the EdgeIIoT dataset, (b): Before balancing the CICIoMT2024 dataset, (c): After balancing the EdgeIIoT dataset, (d): After balancing the CICIoMT2024 dataset.

![](_page_3_Figure_10.jpeg)

Fig. 4. ROC curves of the proposed model. (a): Before balancing EdgeIIoT dataset, (b): Before balancing CICIoMT2024 dataset, (c): After balancing EdgeIIoT dataset, (d): After balancing CICIoMT2024 dataset.

1.00. Overall, data balancing greatly improves classification fairness and accuracy, particularly for harder-to-detect classes. *- Classification report:* Table II summarizes the Pr, Rc, F1 score, and VD per class across two datasets: EdgeIIoTset and CICIoMT2024; before and after data balancing. Before balancing, the model achieved near-perfect results (99–100%) across all classes on EdgeIIoTset, but struggled with minority classes in CICIoMT2024. After balancing, EdgeIIoTset metrics remained consistently high, showing the model's robustness, while CICIoMT2024 exhibited substantial improvements across all metrics, demonstrating better generalization and handling of class imbalance

TABLE II CLASSIFICATION REPORT FOR THE SE ENHANCED-VIT-BILSTM MODEL ON EDGEIIOTSET AND CICIOMT2024 DATASETS.

| Before Data Balancing      |      |      |      |         |  |  |  |  |  |
|----------------------------|------|------|------|---------|--|--|--|--|--|
| EdgeIIoT Dataset           |      |      |      |         |  |  |  |  |  |
|                            | Pr.  | Rc   | F1   | Support |  |  |  |  |  |
| Normal traffic (0)         | 100% | 100% | 100% | 4 985   |  |  |  |  |  |
| DDoS (1)                   | 99%  | 98%  | 99%  | 9 896   |  |  |  |  |  |
| Info. gathering (2)        | 100% | 100% | 100% | 4 333   |  |  |  |  |  |
| MITM (3)                   | 100% | 100% | 100% | 255     |  |  |  |  |  |
| Injection (4)              | 97%  | 98%  | 98%  | 6 083   |  |  |  |  |  |
| Malware (5)                | 100% | 100% | 100% | 6 008   |  |  |  |  |  |
| CICIoMT2024 Dataset        |      |      |      |         |  |  |  |  |  |
| Pr.<br>Rc<br>F1<br>Support |      |      |      |         |  |  |  |  |  |
| Normal traffic             | 99%  | 96%  | 94%  | 6 761   |  |  |  |  |  |
| DDoS UDP flood             | 91%  | 100% | 96%  | 472     |  |  |  |  |  |
| DoS UDP flood              | 60%  | 83%  | 69%  | 455     |  |  |  |  |  |
| MITM                       | 92%  | 86%  | 89%  | 209     |  |  |  |  |  |
| MQTT                       | 92%  | 94%  | 93%  | 181     |  |  |  |  |  |
| Recon                      | 99%  | 99%  | 99%  | 1 650   |  |  |  |  |  |

| After Data Balancing       |      |      |      |         |  |  |  |  |  |
|----------------------------|------|------|------|---------|--|--|--|--|--|
| EdgeIIoT Dataset           |      |      |      |         |  |  |  |  |  |
| Pr.<br>Rc<br>F1<br>Support |      |      |      |         |  |  |  |  |  |
| Normal traffic (0)         | 100% | 100% | 100% | 9 860   |  |  |  |  |  |
| DDoS (1)                   | 99%  | 97%  | 98%  | 10 029  |  |  |  |  |  |
| Info. gathering (2)        | 100% | 100% | 100% | 9 878   |  |  |  |  |  |
| MITM (3)                   | 100% | 100% | 100% | 9 815   |  |  |  |  |  |
| Injection (4)              | 97%  | 99%  | 98%  | 9 769   |  |  |  |  |  |
| Malware (5)                | 100% | 100% | 100% | 9 925   |  |  |  |  |  |
| CICIoMT2024 Dataset        |      |      |      |         |  |  |  |  |  |
|                            | Pr.  | Rc.  | F1   | Support |  |  |  |  |  |
| Normal traffic             | 92%  | 96%  | 94%  | 6 284   |  |  |  |  |  |
| DDoS UDP flood             | 100% | 99%  | 99%  | 6 644   |  |  |  |  |  |
| DoS UDP flood              | 95%  | 93%  | 94%  | 6 598   |  |  |  |  |  |
| MITM                       | 99%  | 100% | 99%  | 6 399   |  |  |  |  |  |
| MQTT                       | 100% | 98%  | 99%  | 6 668   |  |  |  |  |  |
| Recon                      | 100% | 100% | 100% | 6 551   |  |  |  |  |  |

Abbreviations: Validation data (Support), Precision (Pr), Recall (Rc).

*- Comparative study:* Table IV compares the proposed ViT-BiLSTM model with state-of-the-art approaches for multiclass classification on the EdgeIIoT and CICIoMT2024 datasets. The proposed model achieves the highest accuracy on EdgeI-IoT and competitive performance on CICIoMT2024. Although a prior study [23] reports slightly higher accuracy on CI-CIoMT2024, it omits key metrics like FPR and loss. Unlike earlier methods that focus mainly on accuracy, the proposed model delivers comprehensive performance; combining high accuracy, full metric reporting, and low inference time.

#### *D. Ablation Study*

An ablation study (Table III) was conducted using four model variants to assess the impact of different ViT and BiLSTM configurations in both sequential and parallel setups. Across the EdgeIIoT and CICIoMT2024 datasets, Model #3 achieved the best results; 99.33% accuracy on EdgeIIoT and 98.16% on CICIoMT2024; along with low loss and FPR. This highlights the effectiveness of parallel feature extraction and fusion, enabling the model to capture both spatial and temporal patterns efficiently. Increasing the BiLSTM size in Model #4 did not yield better performance, showing that larger models do not always generalize better. Sequential designs (Models #1 and #2) performed reasonably well but were outperformed by the parallel architecture.

### V. CONCLUSION

This study introduced an enhanced SE ViT-BiLSTM-based intrusion detection framework designed to accurately detect a broad range of cyberattacks in both industrial and medical IoT environments. Through rigorous evaluation on two realworld benchmark datasets; EdgeIIoT and CICIoMT2024; the model demonstrated high classification accuracy and real-time performance, both before and after applying class balancing techniques (SMOTE and RandomOverSampler). These results highlight the framework's effectiveness in addressing data imbalance and its suitability for critical IoT applications. Looking ahead, future work will focus on improving the model's generalizability by testing it on additional datasets, optimizing it for deployment through model compression and edge computing, and enhancing interpretability using explainable AI (XAI) tools to support human decision-making in security operations.

## REFERENCES

- [1] K. Kamir and C. Sarra, "Machine learning solutions for securing iotbased healthcare: A review," in *2023 5th International Conference on Pattern Analysis and Intelligent Systems (PAIS)*. IEEE, 2023, pp. 1–8.
- [2] C. Ni and S. C. Li, "Machine learning enabled industrial iot security: Challenges, trends and solutions," *Journal of Industrial Information Integration*, vol. 38, p. 100549, 2024.
- [3] A. Gueriani, H. Kheddar, and A. C. Mazari, "Cyber threat detection in iiot and iomt using dnn-gru with multi-head attention," in *2025 International Conference on Research in Computing at Feminine (RIF)*. IEEE, 2025, pp. 1–8.
- [4] ——, "Explainable bilstm-mha-based ids for iot using shap and zeroday attack detection," in *2025 International Conference on Artificial Intelligence and Innovative Applications (AIIA)*. IEEE, 2025, pp. 1–8.
- [5] S. F. Ahmed, M. S. B. Alam, S. Afrin, S. J. Rafa, N. Rafa, and A. H. Gandomi, "Insights into internet of medical things (iomt): Data fusion, security issues and potential solutions," *Information Fusion*, vol. 102, p. 102060, 2024.
- [6] F. Xu, S. Liu, and X. Yang, "An efficient privacy-preserving authentication scheme with enhanced security for iomt applications," *Computer Communications*, vol. 208, pp. 171–178, 2023.
- [7] M. Wang, N. Yang, and N. Weng, "Securing a smart home with a transformer-based iot intrusion detection system," *Electronics*, vol. 12, no. 9, p. 2100, 2023.
- [8] L. Sana, M. M. Nazir, J. Yang, L. Hussain, Y.-L. Chen, C. S. Ku, M. Alatiyyah, and L. Y. Por, "Securing the iot cyber environment: Enhancing intrusion anomaly detection with vision transformers," *IEEE Access*, 2024.
- [9] B. Buyuktanir, S¸. Altinkaya, G. Karatas Baydogmus, and K. Yildiz, "Federated learning in intrusion detection: advancements, applications, and future directions," *Cluster Computing*, vol. 28, no. 7, pp. 1–25, 2025.
- [10] V. P. Gandi, N. S. L. Jatla, G. Sadhineni, S. Geddamuri, G. K. Chaitanya, and A. Velmurugan, "A comparative study of ai algorithms for anomalybased intrusion detection," in *2023 7th International Conference on Computing Methodologies and Communication (ICCMC)*. IEEE, 2023, pp. 530–534.

TABLE III
PERFORMANCE OF DIFFERENT VARIANTS OF THE PROPOSED MODELS IN MULTICLASS CLASSIFICATION.

| Model |                                | Description                                        | EdgeIIoT |          |         | CICIoMT2024 |          |         |
|-------|--------------------------------|----------------------------------------------------|----------|----------|---------|-------------|----------|---------|
|       | Wiodei                         | Description                                        | Acc. (%) | Loss (%) | FPR (%) | Acc. (%)    | Loss (%) | FPR (%) |
| #1    | $ViT \rightarrow BiLSTM_{32}$  | ViT features passed to a BiLSTM sequentially       | 98.99    | 0.0281   | 0.0023  | 97.10       | 0.0799   | 0.0100  |
| #2    | BiLSTM $_{32} \rightarrow ViT$ | BiLSTM features passed to a ViT sequentially       | 98.93    | 0.0286   | 0.0023  | 94.64       | 0.1186   | 0.0106  |
| #3    | ViT    BiLSTM 32               | ViT and BiLSTM $_{32}$ run in parallel, then fused | 99.33    | 0.0158   | 0.0013  | 98.16       | 0.0578   | 0.0036  |
| #4    | ViT    BiLSTM 64               | ViT and BiLSTM 64 run in parallel, then fused      | 98.98    | 0.0301   | 0.0040  | 96.01       | 0.0898   | 0.005   |

#### TABLE IV

PERFORMANCE METRICS OF THE PROPOSED ENHANCED VIT-BILSTM MODEL IN COMPARISON TO STATE-OF-THE-ART METHODS FOR MULTICLASS CLASSIFICATION.

| Work | Model                  | Dataset  | Acc    | Loss   | Pr    | Rc    | F1    | FPR    | Inf<br>time |
|------|------------------------|----------|--------|--------|-------|-------|-------|--------|-------------|
| [24] | Generic CNN            | EdgeIIoT | 98.98  | Х      | X     | Х     | Х     | Х      | Х           |
| [25] | LSTM/DNN               | CICIoMT  | 79     | X      | 78    | 79    | 76    | X      | X           |
| [26] | CNN-LSTM-<br>ResNet-SA | EdgeIIoM | I33.30 | Х      | 33.31 | 100   | 49.97 | Х      | Х           |
|      |                        | CICIoT   | 99.88  | X      | 99.89 | 99.99 | 99.94 | X      | X           |
| [23] | CNN                    | EdgeIIoT | 96.50  | X      | 97.48 | 96.50 | 96.41 | X      | 0.00012     |
| [20] |                        | CICIoMT  | 99.67  | X      | 99.67 | 99.67 | 99.66 | X      | 0.00004     |
| Our  | ViT-BiLSTM             | EdgeIIoT | 99.33  | 0.0158 | 99.33 | 99.33 | 99.33 | 0.0013 | 0.00035     |
|      |                        | CICIoMT  | 98.16  | 0.0578 | 98.16 | 98.16 | 98.15 | 0.0036 | 0.00014     |

Acc. Pr. Rc. and FPR are in (%). Inference time is in (seconds/instance).

- [11] A. Ali, A. W. Septyanto, I. Chaudhary, H. Al Hamadi, H. M. Alzoubi, and Z. F. Khan, "Applied artificial intelligence as event horizon of cyber security," in 2022 International Conference on Business Analytics for Technology and Security (ICBATS). IEEE, 2022, pp. 1–7.
- [12] O. Ceran, E. Özdoğan, and M. Uysal, "Leveraging graph neural networks for iot attack detection," Sakarya University Journal of Computer and Information Sciences, vol. 8, no. 2, pp. 223–244, 2025.
- [13] H. Peng, C. Wu, and Y. Xiao, "A bilstm-based iot intrusion detection system with mutual information and focal loss," in 2024 6th International Conference on Frontier Technologies of Information and Computer (ICFTIC). IEEE, 2024, pp. 1–6.
- [14] A. Gueriani, H. Kheddar, A. C. Mazari, and M. C. Ghanem, "A robust cross-domain ids using bigru-lstm-attention for medical and industrial iot security," arXiv preprint arXiv:2508.12470, 2025.
- [15] A. Gueriani, H. Kheddar, and A. C. Mazari, "Hybrid resnet-1d-bigru with multi-head attention for cyberattack detection in industrial iot environments," in 2025 International Conference on Intelligent Computer Systems, Data Science and Applications (IC2SDA). IEEE, 2025, pp. 1–6.

- [16] M. A. Ferrag, O. Friha, D. Hamouda, L. Maglaras, and H. Janicke, "Edge-iiotset: A new comprehensive realistic cyber security dataset of iot and iiot applications for centralized and federated learning," *IEEE Access*, vol. 10, pp. 40281–40306, 2022.
- [17] S. Dadkhah, E. C. P. Neto, R. Ferreira, R. C. Molokwu, S. Sadeghi, and A. A. Ghorbani, "Ciciomt2024: A benchmark dataset for multi-protocol security assessment in iomt," *Internet of Things*, vol. 28, p. 101351, 2024
- [18] A. Gueriani, H. Kheddar, and A. C. Mazari, "Deep reinforcement learning for intrusion detection in IoT: A survey," in 2023 2nd International Conference on Electronics, Energy and Measurement (IC2EM), vol. 1. IEEE, 2023, pp. 1–7.
- [19] H. Kheddar, M. Hemis, Y. Himeur, D. Megías, and A. Amira, "Deep learning for steganalysis of diverse data types: A review of methods, taxonomy, challenges and future directions," *Neurocomputing*, p. 127528, 2024
- [20] H. Kheddar, Y. Himeur, S. Al-Maadeed, A. Amira, and F. Bensaali, "Deep transfer learning for automatic speech recognition: Towards better generalization," *Knowledge-Based Systems*, vol. 277, p. 110851, 2023.
- [21] A. Gueriani, H. Kheddar, and A. C. Mazari, "Adaptive cyber-attack detection in iiot using attention-based lstm-cnn models," in 2024 International Conference on Telecommunications and Intelligent Systems (ICTIS). IEEE, 2024, pp. 1–6.
- [22] ——, "Enhancing iot security with cnn and lstm-based intrusion detection systems," in 2024 6th International Conference on Pattern Analysis and Intelligent Systems (PAIS). IEEE, 2024, pp. 1–7.
- [23] K. Kharoubi, S. Cherbal, D. Mechta, and A. Gawanmeh, "Network intrusion detection system using convolutional neural networks: Nidsdl-cnn for iot security," *Cluster Computing*, vol. 28, no. 4, p. 219, 2025.
- [24] M. Singh and N. Chauhan, "Convolutional neural network based iot intrusion detection system using edge-iiotset," in 2024 International Conference on Integrated Circuits, Communication, and Computing Systems (ICIC3S), vol. 1. IEEE, 2024, pp. 1–4.
- [25] N. C. Kavkas and K. Yildiz, "Enhancing lomt security with deep learning based approach for medical iot threat detection," in 2025 13th International Symposium on Digital Forensics and Security (ISDFS). IEEE, 2025, pp. 1–5.
- [26] T. Sasi, A. H. Lashkari, R. Lu, P. Xiong, and S. Iqbal, "An efficient self attention-based 1d-cnn-lstm network for iot attack detection and identification using network traffic," *Journal of Information and Intelligence*, 2024.