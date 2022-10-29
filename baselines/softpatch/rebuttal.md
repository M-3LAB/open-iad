#Rebuttal
#The Response to Reviewer m85i (Part 1)
Thanks for your valuable comments. We will answer the questions one by one.
####Q1: Methodology novelty.
We agree that our anomaly detection structure is based on the proven work: PatchCore. However, we would highlight that this is the first work on noisy label learning in unsupervised sensory anomaly detection. As stated in our manuscript, we think noisy label learning could be a new paradigm of anomaly detection, which motivates us to solve it by referring to some existing techniques. Despite the novelty and significance of the problem, SoftCore is not a direct application of the current denoising algorithm. Traditional label noise methods are designed for supervised methods, and data cleaning methods are not efficient for "sensory anomaly detection" [1], where the gap between the abnormal and normal images is tiny. In brief, we elegantly denoise in patch level and reconstruct the memory bank with soft boundary so that the robustness is achieved, and such a reformulation is nontrivial. The comparative experiments on image-level denoising and patch-level denoising in the appendix(Supplementary Material A.4 and A.5) can also illustrate the advantage of denoising at the patch level.
####Q2: The setting of 'Overlap' experiment.
Thank you for raising an interesting point here. The overlap setting here is to simulate exactly the same defects in the product due to structural problems in actual production. Due to the limitation that there are almost no defects with the same appearance at the same position in the MVTecAD dataset, an overlap setting is provided as a reference. In addition, we did another experiment where the overlap images were augmented (blur and noise) in the train set to enhance fairness. The result below shows that our method still presents good robustness when the overlap samples have been transformed. 

|    Setting    | Overlap with guassian   noise |                | Overlap with noise and   blur |                | Overlap with rotation |                | Overlap with affine   transformation |                |
|:-------------:|:-----------------------------:|:--------------:|:-----------------------------:|:--------------:|:---------------------:|:--------------:|:------------------------------------:|:--------------:|
|     Method    |           PatchCore           | SoftCore(ours) |           PatchCore           | SoftCore(ours) |       PatchCore       | SoftCore(ours) |               PatchCore              | SoftCore(ours) |
|   Detection   |             0.760             |   **0.984 **   |             0.848             |   **0.984 **   |         0.950         |   **0.984 **   |                 0.933                |   **0.984 **   |
| Localization  |             0.790             |   **0.969 **   |             0.864             |   **0.970 **   |         0.924         |   **0.978 **   |                0.915                 |   **0.978 **   |

####Q3: The small performance advantage compared with state-of-the-art methods in 'no overlap' setting. 
Thank you for raising an interesting point here. First, the little performance drops of existing methods in the noisy setting are an interesting phenomenon. As mentioned in Q2, the same defects are rare in the MVTecAD dataset. So existing methods trained in a part of anomaly data can still detect most other anomalies with a different appearance. Despite this, our method presents better noise robustness and has a clear lead(0.7) in anomaly location performance. An additional experiment in a more diverse dataset, BTAD [2], further shows the advantages of our approach. The result is shown below; attention that this is the result of anomaly detection under the noise-free condition. By screening train samples, we find that there is already some noise(usually small scratch) in the category BTAD_02, which further demonstrates the necessity of our approach. Meanwhile, more anomaly samples increase the probability of similar appearance anomalies. 

| category | PatchCore | PaDiM | SoftCore(ours) | Anomaly samples |
|:--------:|:---------:|:-----:|:--------------:|:---------------:|
|  BTAD_01 |   1.000   | 1.000 |      0.999     |        50       |
|  BTAD_02 |   0.871   | 0.871 |    **0.934**   |       200       |
|  BTAD_03 |   0.999   | 0.971 |      0.997     |        41       |
|   Mean   |   0.957   | 0.947 |    **0.977**   |        -        |

####Q4: The analysis of the neighbor's influence is needed. 
PatchCore uses a scaling anomaly score $s^*$ to account for the behavior of neighboring patches. If the nearest neighbor is an outlier in the test, PatchCore improves its robustness by searching more near points. However, this robustness operation is too weak in our experiment with noisy data. The soft weights in our method are from patch-level noise discrimination, which gives a higher scaling weight to the patch far from the cluster. So the soft weights have already considered the local relationship. 
####Q5: Confusing notion and sentence.
Thanks, the ’N, H, W, C’ in Fig.2 respectively donate number, height, weight and channels of images. We will modify the notations in figures to be consistent with those in formulas. Thanks for pointing out the confusing sentence, we will improve the English statement. 
Some of the above experiments and statements will be added to the revised version of the paper. 

[1] Yang J, Zhou K, Li Y, et al. Generalized out-of-distribution detection: A survey[J]. arXiv preprint arXiv:2110.11334, 2021.

[2] Mishra P, Verk R, Fornasier D, et al. VT-ADL: A vision transformer network for image anomaly detection and localization[C]//2021 IEEE 30th International Symposium on Industrial Electronics (ISIE). IEEE, 2021: 01-06.

#The Response to Reviewer MPDh
Thanks for your valuable comments. We will answer the questions one by one.
####Q1: The idea of reweighting the noisy training data and using LOF is not new.
As mentioned in the references you listed, reweighting training data [1] is a common technique to handle noisy labels, and LOF [2] has been well developed in outlier detection. However, we would highlight that we novelty apply these methods in building a patch-level memory bank. This is from a motivation that an anomaly image contains not only defective areas but also large areas of normal. The patch-level denoising strategy improves the data usage rate compared to conventional sample-level denoising in the proposed task. The comparative experiments on image-level denoising and patch-level denoising in the appendix (Supplementary Material A.4 and A.5) can also illustrate the advantage of denoising at the patch level. Meanwhile, the mentioned MemAE [3] uses a memory module for reconstruction and weights for addressing, which is far different from our approach. 
####Q2: There are some existing studies of anomaly detection with noisy data. 
Thank you for raising an interesting point here. The papers you mentioned research the robustness of different objects from us. For example, the references [4, 5] respectively deal with video and tabular data. The paper [6] focuses on semantic anomaly detection where the anomaly is several classes in a multi-class data set, such as MNIST in this paper. Unlike semantic anomaly detection, we focus on sensory anomaly detection [7], which has recently raised much concern. In sensory anomaly detection, the difference between anomaly and normal is tiny, and anomaly location is needed. Visual sensory anomaly detection is mainly used in real industrial scenarios [8], and the semantic methods have poor performance on it [9]. A more precise way to say it is "we are the first one to study the visual sensory anomaly detection with noisy data." We will update the related parts in the revised version. 
####Q3: Confusion about hyperparameters $\tau$.
The hyperparameters $\tau$ is not specifically set according to the noise ratio. In fact, we use a conservative value to get the result. As shown in Figure.4, the performance is stable when the hyperparameters $\tau$ is larger than 0.09. Another experiment in Appendix.4 shows the performance trends in different noise ratios where SoftCore still has outperformance. In addition, the rate of defective products in real industrial production is often under an acceptable level, so the noise ratio would not beyond expectation. 
####Q4: Limitations are not thoroughly discussed. 


code

[1] Liu, Tongliang and Dacheng Tao. “Classification with Noisy Labels by Importance Reweighting.” IEEE Transactions on Pattern Analysis and Machine Intelligence 38 (2016): 447-461.

[2] Alghushairy, Omar, Raed Alsini, Terence Soule and Xiaogang Ma. “A Review of Local Outlier Factor Algorithms for Outlier Detection in Big Data Streams.” Big Data Cogn. Comput. 5 (2021): 1.

[3] Gong, Dong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh and Anton van den Hengel. “Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection.” 2019 IEEE/CVF International Conference on Computer Vision (ICCV) (2019): 1705-1714.

[4] Pang, Guansong, Cheng Yan, Chunhua Shen, Anton van den Hengel and Xiao Bai. “Self-Trained Deep Ordinal Regression for End-to-End Video Anomaly Detection.” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2020): 12170-12179.

[5] Liu, Boyang, Ding Wang, Kaixiang Lin, Pang-Ning Tan and Jiayu Zhou. “RCA: A Deep Collaborative Autoencoder Approach for Anomaly Detection.” IJCAI : proceedings of the conference 2021 (2021): 1505-1511 .

[6] Zhou, Chong and Randy Clinton Paffenroth. “Anomaly Detection with Robust Deep Autoencoders.” Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2017): n. pag.

[7] Yang J, Zhou K, Li Y, et al. Generalized out-of-distribution detection: A survey[J]. arXiv preprint arXiv:2110.11334, 2021.

[8] Tao X, Gong X, Zhang X, et al. Deep Learning for Unsupervised Anomaly Localization in Industrial Images: A Survey[J]. arXiv e-prints, 2022: arXiv: 2207.10298.

[9] Bergmann P, Fauser M, Sattlegger D, et al. MVTec AD--A comprehensive real-world dataset for unsupervised anomaly detection[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 9592-9600.


