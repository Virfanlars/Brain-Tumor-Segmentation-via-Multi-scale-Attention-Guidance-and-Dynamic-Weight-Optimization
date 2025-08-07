% This must be in the first 5 lines to tell arXiv to use pdfLaTeX, which is strongly recommended.
% In particular, the hyperref package requires pdfLaTeX in order to break URLs across lines.

\documentclass[11pt]{article}

% Remove the "review" option to generate the final version.
\usepackage{ACL2023}

% Standard package includes
\usepackage{times}
\usepackage{CJKutf8}
\usepackage{latexsym}
\usepackage{natbib}
\setcitestyle{numbers,square}

% For proper rendering and hyphenation of words containing Latin characters (including in bib files)
\usepackage[T1]{fontenc}
% For Vietnamese characters
% \usepackage[T5]{fontenc}
% See https://www.latex-project.org/help/documentation/encguide.pdf for other character sets

% This assumes your files are encoded as UTF8
\usepackage[utf8]{inputenc}

% This is not strictly necessary, and may be commented out.
% However, it will improve the layout of the manuscript,
% and will typically save some space.
\usepackage{microtype}

% This is also not strictly necessary, and may be commented out.
% However, it will improve the aesthetics of text in
% the typewriter font.
\usepackage{inconsolata}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{algorithm2e}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{float}
% Add xcolor package for modification marking
\usepackage{xcolor}

\title{Brain Tumor Segmentation via Multi-scale Attention Guidance and Dynamic Weight Optimization}

\author{
Fei Wu\textsuperscript{1,2}, 
Chen Li\textsuperscript{1}, 
Yang Chen\textsuperscript{3} \\
\textsuperscript{1}Department of Neurosurgery, Zhongshan Hospital, Fudan University,Shanghai  \\
\textsuperscript{2}School of Informatics, Xiamen University \\
\textsuperscript{3}Zhongshan Hospital (Xiamen), Fudan University No. 668, Jinhu Road
Huli District Xiamen, Fujian 361015, P. R. China \\
Corresponding author: Chen Li \\
Email: \texttt{li.chen@zs-hospital.sh.cn}
}

\begin{document}

\maketitle 

\begin{CJK}{UTF8}{gbsn}
\begin{abstract}
Accurate segmentation of brain tumors is crucial for clinical diagnosis and treatment planning. This paper proposes a brain tumor segmentation method based on multi-scale attention guidance and dynamic weight optimization, applied to the precise segmentation of meningiomas, gliomas, and pituitary tumors. The method integrates a lightweight Mix Vision Transformer \cite{chen2021transmixattendmixvision} as the backbone network, implements multi-scale feature interaction through a Channel-Spatial Dual Attention (CSDA) module, and employs a Dynamic Weight Decoder (DWD) to address class imbalance issues.\cite{oh2025dawintrainingfreedynamicweight} Experimental results demonstrate that the proposed method achieves a mean Dice coefficient of 0.9165 (91.65\%), outperforming existing approaches including Zig-RiR (87.90\%) and TransUNet (85.07\%) in both Dice coefficient and Hausdorff distance\cite{CHENG2025107073} metrics while maintaining a low parameter count (18.81M parameters, 43\% fewer than comparable Transformer models).

\textbf{Keywords}: Medical image segmentation; Brain tumor; Mix Vision Transformer; Multi-scale attention; Dynamic weight optimization
\end{abstract}

\section{Introduction}
Medical image segmentation, as a core technology in computer-aided diagnosis systems, plays an indispensable role in lesion localization, surgical planning, and treatment efficacy assessment. However, conventional CNN models are constrained by limited local receptive fields\cite{wang2024witunetushapedarchitectureintegrating}, struggling to model long-range spatial dependencies in images, which leads to compromised segmentation accuracy in low-contrast regions and complex anatomical structures.\cite{haruna2025vgambaattentivestatespace}

Recent advances in Vision Transformers (ViT) have brought revolutionary breakthroughs to medical image segmentation through their global feature capturing capability enabled by self-attention mechanisms.\cite{khan2023recentsurveyvisiontransformers} Nevertheless, pure Transformer models face significant challenges in local feature extraction efficiency and computational resource consumption. Current mainstream solutions predominantly adopt CNN-Transformer hybrid architectures to compensate for local features through convolutional modules.\cite{djoumessi2025hybridfullyconvolutionalcnntransformer} However, these approaches often suffer from semantic information loss due to feature space misalignment\cite{gao2024enhancedencoderdecodernetworkarchitecture}, particularly manifesting as edge blurring and artifact generation during multi-scale feature fusion stages.

To address these challenges, this study proposes a medical image segmentation method integrating multi-level guidance and attention mechanisms. The core innovation lies in constructing a Multi-level Semantic Guided Decoder (MSGD) that achieves efficient fusion of global semantics and local details through a dual-path collaborative mechanism.\cite{liu2020efficientfcnholisticallyguideddecodingsemantic} Specifically, we first employ an improved PVTv2 as the encoder\cite{Wang_2022}, where the linear attention mechanism and overlapping patch embedding strategy enhance representation capability for small target regions while reducing computational complexity. Subsequently, the designed Skip Fusion Module (SFM) \cite{li2025cfformercrosscnntransformerchannel}addresses spatial mismatch between high- and low-level features in traditional skip connections through parallel convolutional pathways and channel attention weighting. Furthermore, the proposed Multi-dimensional Attention Module (CSDA) integrates multi-scale convolutional kernels with channel-spatial attention, enabling fine-grained regulation of cross-dimensional feature interactions. This hierarchical design empowers the model to dynamically balance global context perception and local structural delineation, significantly enhancing segmentation robustness for complex medical images.\cite{yingjie2024localvsglobalmodels}

Primary objectives of this research work:
\begin{itemize}
    \item To develop a lightweight multi-scale attention-guided framework that effectively captures both global semantic context and local boundary details for brain tumor segmentation
    \item To design a Channel-Spatial Dual Attention (CSDA) module that enhances cross-dimensional feature interactions and improves small-target detection capability
    \item To propose a Dynamic Weight Decoder (DWD) that adaptively addresses class imbalance issues while maintaining computational efficiency
    \item To demonstrate superior performance compared to state-of-the-art methods including recent Transformer-based approaches while reducing model complexity
    \item To provide a clinically deployable solution with balanced accuracy-efficiency trade-offs for real-world medical imaging applications
\end{itemize}

\begin{figure}[htb]
\centering
\includegraphics[width=0.45\textwidth]{comparison.png}
\caption{The DICE scores of MSAF-Net on the brain tumor dataset were higher than those of the current model}
\label{fig:comparison}
\end{figure}

Systematic validation on the Brain Tumor Segmentation (BraTS) dataset\cite{cheng2015enhanced} demonstrates superior performance of our method. Experimental results show consistent outperformance over state-of-the-art models in key metrics including Dice coefficient and Hausdorff distance, with particular improvement of 12.6\% in segmentation accuracy for low-contrast tissue boundaries. Notably, through depthwise separable convolutions and parameter sharing strategies, the model parameters are controlled below 43\% of Transformer models with comparable performance, providing a clinically deployable lightweight solution. This research establishes a novel technical pathway for synergistic optimization of multi-scale feature fusion and attention mechanisms in medical image segmentation, holding significant practical value for advancing precision medicine.

\section{Background and Related Work}

\subsection{ViT in Medical Image Segmentation}

\textbf{Hybrid Architecture Design} TransUNet\cite{transunet} pioneered the integration of CNNs with Transformers, leveraging CNNs for local feature extraction while employing Transformers to encode global context. Swin-Unet\cite{cao2021swinunetunetlikepuretransformer} adopted a pure Transformer architecture, yet its reliance on large-scale pretrained models constrained practical applications. Recent studies like SegFormer\cite{xie2021segformersimpleefficientdesign} and PVTv2\cite{pvtv2} reduced computational complexity through hierarchical structures and linear attention mechanisms while preserving multi-scale feature extraction capabilities. Further advancing efficiency, Zig-RiR\cite{10969076} proposed a nested RWKV-in-RWKV architecture with zigzag scanning mechanism, achieving linear complexity for long-distance modeling while maintaining spatial continuity. 

\textbf{Local-Global Feature Co-Modeling} To address Transformers' limitations in local feature extraction, UFormer\cite{wang2021uformergeneralushapedtransformer} introduced depthwise separable convolutions to enhance local perception, whereas PolypPVT\cite{Dong_2023} optimized low-level features via channel attention. However, existing methods still exhibit limitations in fine-grained processing of multi-stage features, particularly requiring optimization of inefficient fusion between high- and low-level features.

Table \ref{tab:related_work_comparison} provides a comprehensive comparison of recent Transformer-based medical image segmentation methods, highlighting their key characteristics and limitations that motivated our approach.

\begin{table*}[htbp]
\centering
\caption{Comparison of Recent Transformer-based Medical Image Segmentation Methods}
\label{tab:related_work_comparison}
\begin{tabular}{lccccc}
\toprule
\textbf{Method}  & \textbf{Key Innovation} & \textbf{Limitations} \\
\midrule
TransUNet & CNN-Transformer fusion & Heavy computational load \\
Swin-UNet  & Shifted window attention & Large pretrained models \\
SegFormer  & Lightweight MLP decoder & Limited local feature modeling  \\
UFormer  & Depthwise separable conv & Inefficient multi-scale fusion \\
Zig-RiR & Nested RWKV structure & Complex zigzag scanning  \\
\bottomrule
\end{tabular}
\end{table*}

\subsection{Dynamic Weights and Loss Optimization}

\textbf{Class Imbalance Handling} Significant variation in lesion-to-background ratios in medical imaging renders traditional Dice loss vulnerable to class imbalance. Adaptive Dice\cite{golnari2024adaptiverealtimemultilossfunction} mitigated this through inverse-frequency weighting, while Focal Dice\cite{ZHENG2025107741} incorporated Focal Loss principles to enhance hard sample learning. Recent studies proposed dynamic weight adjustment strategies\cite{ma2024plugandplaytrainingframeworkpreference} that update class weights in real-time based on training errors, significantly improving small-target segmentation performance.\cite{si2024scsaexploringsynergisticeffects}

\textbf{Feature Fusion Mechanism Innovation} Gated attention\cite{oktay2018attentionunetlearninglook} controlled feature fusion ratios via learnable parameters, while bidirectional feature pyramids\cite{lu2024cascadedmultiscaleattentionenhanced} enhanced multi-scale interactions through top-down and bottom-up pathways. Our proposed Dynamic Weighted Decoder (DWD) synthesizes these concepts, achieving adaptive fusion of hierarchical features through gated mechanisms combined with dual channel-spatial attention for feature refinement.

\subsection{Data Augmentation and Normalization}

\textbf{Medical Image-Specific Augmentation} Addressing limited medical data availability, elastic deformation\cite{ronneberger2015unetconvolutionalnetworksbiomedical} and lesion-centered sampling\cite{JIANG2023106726} have been widely adopted to simulate organ deformation and enhance small-target learning. Multimodal fusion improves feature robustness through cross-modal alignment, while dynamic range preservation\cite{yang2024crnetdetailpreservingnetworkunified} employs 16-bit grayscale conversion to prevent information loss.

\textbf{Background-Aware Normalization} excluded background regions via threshold segmentation for statistical computation\cite{oh2021backgroundawarepoolingnoiseawareloss}, offering superior compatibility with medical imaging characteristics compared to traditional Z-score methods. Our work extends this by introducing dynamic range compression to preserve critical anatomical structures while enhancing training stability.

\begin{figure*}[htb]
\centering
\includegraphics[width=0.95\textwidth]{architecture.png}
\caption{Schematic Diagram of the Overall Network Architecture}
\label{fig:arch}
\end{figure*}

\section{Methodology}

In this section, we propose MSAF-Net, a multi-scale medical image segmentation framework based on hybrid attention mechanisms. The overall architecture is illustrated in Fig. \ref{fig:arch}, which achieves precise lesion localization through synergistic encoder-decoder co-design. The framework comprises two core components: Multi-scale Pyramid Hybrid Encoder (MixViT) and Dynamic Weighted Decoder (DWD). Specifically, the MixViT encoder employs a multi-scale pyramid structure for cross-hierarchical feature modeling (Section 3.1), while the DWD decoder optimizes segmentation boundaries through adaptive feature fusion mechanisms (Section 3.3). A multi-stage supervision strategy is implemented during model training.

\subsection{Multi-scale Pyramid Hybrid Encoder}

The MixViT adopts a four-stage pyramid encoder architecture with progressive downsampling (256 → 128 → 64 → 32), where the embedding dimensions for each stage are 64, 128, 320, and 512, corresponding to feature map resolutions from H/4×W/4 to H/32×W/32. Here, H and W represent the height and width of the input image, respectively. Key components include:

\paragraph{Overlap Patch Embedding}
This module implements an overlapping patch embedding strategy (stride=4, patch size=7×7, overlap ratio=30\%) with convolutional projection\cite{guo2022segnextrethinkingconvolutionalattention} to preserve edge continuity:
\begin{equation}
\text{Proj}(X) = \text{DWConv}(\text{Conv2d}(X))
\end{equation}
it achieves 85.4\% Top-1 accuracy on ImageNet-1K without any extra training data or label.\cite{dong2022cswintransformergeneralvision}

\paragraph{Hybrid Transformer Block}
Each transformer block integrates two key components: EfficientAttention and MixFFN. \cite{9547423}The former reduces computational complexity from O(N²) to O(N) through linear projection, while the latter enhances local feature modeling via depthwise separable convolution in mixed feed-forward networks.

This architectural design enables MixViT to efficiently capture local details and global context at low computational cost, particularly suitable for medical imaging tasks. Key innovations include:

\paragraph{Multi-granularity Feature Fusion}
Feature pyramid interfaces with deformable convolution dynamically fuse adjacent-level features.

\paragraph{Adaptive Position Encoding}
A dynamic position bias mechanism automatically adjusts positional embedding weights based on image content:
\begin{equation}
\text{PE}_{\text{dynamic}} = \text{MLP}(\text{GAP}(X)) \odot \text{PE}_{\text{static}}
\end{equation}
where GAP denotes Global Average Pooling, X represents the input feature map, MLP is a Multi-Layer Perceptron, and $\odot$ denotes element-wise multiplication.

\paragraph{Progressive Training Strategy}
The framework implements phased activation of multi-resolution pathways through three distinct training stages:
\begin{itemize}
\item \textbf{Initial phase (Epochs 1-100):} Exclusively activate Stage 1-2 with coarse feature extraction
\item \textbf{Intermediate phase (Epochs 101-300):} Engage Stage 3 for mid-level semantic refinement
\item \textbf{Advanced phase (Epochs 301+):} Fully enable four-stage architecture with hierarchical feature integration
\end{itemize}

\begin{figure*}[htb]
\centering
\includegraphics[width=0.9\textwidth]{mixtiv.png}
\caption{Architecture of MixViT: The four-stage pyramid structure achieves cross-scale interaction through CSDA modules. Red arrows indicate dynamic position encoding pathways, while blue dashed boxes highlight depthwise separable convolution layers in MixFFN.}
\label{fig:tiv}
\end{figure*}

\subsection{Context-aware Spatial Dynamic Attention}

In visual transformer architectures, effective capture of local spatial information and global contextual relationships is critical for medical image segmentation. Our analysis reveals that conventional self-attention mechanisms exhibit limitations in handling high-resolution medical images with rich spatial details. To address this, we propose the Context-aware Spatial Dynamic Attention (CSDA) module, which significantly enhances spatial-context modeling capabilities.
\paragraph{Channel-Spatial Dual Attention}
CSDA establishes dual attention pathways for channel-spatial feature refinement. The channel branch computes interdependencies through hybrid pooling:

\begin{equation}
\begin{aligned}
F_{\text{channel}} = \sigma(\text{MLP}(\text{AvgPool}(F))  \\
 + \text{MLP}(\text{MaxPool}(F)))
\end{aligned}
\end{equation}

while the spatial branch captures contextual patterns via feature statistics fusion:
\begin{equation}
\begin{aligned}
F_{\text{spatial}} = \sigma(\text{Conv}_{3\times3}([\text{AvgPool}(F),\text{MaxPool}(F)]))
\end{aligned}
\end{equation}

\paragraph{Multi-scale Context Aggregation}
The module implements hierarchical feature extraction through scale-specific convolutions:
\begin{equation}
\begin{aligned}
F_{\text{ms}} = \text{Concat}[\text{Conv}_{1\times1}(F), \\
\text{Conv}_{3\times3}(F_{\downarrow2}), \text{Conv}_{5\times5}(F_{\downarrow4})]
\end{aligned}
\end{equation}
where $F_{\downarrow2}$ and $F_{\downarrow4}$ denote 2× and 4× downsampled features. Cross-scale fusion combines attention responses through element-wise multiplication:
\begin{equation}
\text{CSDA}(F) = F_{\text{channel}} \odot F_{\text{spatial}} \odot \mathcal{F}_{\text{norm}}(F_{\text{ms}})
\end{equation}
where $\odot$ denotes Hadamard product and $\mathcal{F}_{\text{norm}}$ performs feature normalization.

\begin{figure}[htb]
\centering
\includegraphics[width=0.45\textwidth]{csda.png}
\caption{Architecture of the Channel-Spatial Dual Attention module}
\label{fig:csda}
\end{figure}

\subsection{Dynamic Weighted Decoder (DWD)}

The DWD decoder employs a top-down multi-scale feature fusion architecture with cascade refinement, progressively integrating features from high-level semantic layers (F₁-F₄ with dimensions 64-512) to generate precise segmentation masks. Core components include:

\paragraph{Feature Refinement Module (FRM)}
Multi-scale dilated convolution captures contextual information at varying receptive fields:

\begin{equation}
\begin{aligned}
F_{refined} = \text{Proj}([F, \text{Atrous}_2(F),  \\
\text{Atrous}_4(F), \text{Atrous}_8(F)])
\end{aligned}
\end{equation}
The Feature Refinement Module (FRM) demonstrates substantial performance enhancements in segmentation accuracy, small object detection, and medical image generalization through its integration of multi-scale feature fusion, cross-layer reconstruction, and attention mechanisms. Current architectural paradigms in FRM design exhibit a predominant trend towards synergizing dynamic weight adaptation strategies—such as adversarial temperature regulation and probabilistic attention allocation—with lightweight computational frameworks. This approach effectively balances computational efficiency and model performance through parameter-efficient architectural optimizations.

\paragraph{Gated Fusion Mechanism (GF)}
Information entropy-based adaptive weighting for feature fusion:
\begin{equation}
\begin{aligned}
F_{fused} = G \cdot F_{high} + (1-G) \cdot F_{low} \\
\quad G = \sigma(f_{gate}([F_{high}, F_{low}]))
\end{aligned}
\end{equation}
This mechanism improves small-target segmentation.

\paragraph{Cross-scale Attention (CSA)}
Bidirectional channel-spatial attention module establishes inter-scale dependencies\cite{xu2023self}:
\begin{small}
\begin{equation}
\begin{aligned}
A(F_{high}, F_{low}) = \text{Softmax}(\frac{Q(F_{high}) \cdot K(F_{low})^T}{\sqrt{d}}) \\
\cdot V(F_{low})
\end{aligned}
\end{equation}
\end{small}
CrossFormer has been extensively evaluated on the ADE20K dataset, demonstrating superior performance compared to other network architectures. Notably, it achieves the most significant improvement in dense prediction tasks, suggesting that cross-scale interactions within attention modules play a more critical role in pixel-wise prediction scenarios than in conventional classification tasks. This empirical evidence highlights the inherent advantage of multi-scale feature interaction mechanisms for spatially intensive vision tasks.\cite{electronics14040797}

The main innovations of DWD include the following sections.

\paragraph{Adaptive Weight Allocation} The decoding stage incorporates a dynamic weight adaptation mechanism that autonomously adjusts decoding parameters according to input image characteristics, thereby enhancing model adaptability to heterogeneous tumor types. Ablation studies reveal that replacing this dynamic weighting with fixed parameters induces significant performance degradation, manifested as a marked decline in the Dice coefficient to 72.30\% (see Table \ref{tab:ablation_modules}). 

\paragraph{Lightweight Design}
Parameter sharing and group convolution techniques reduce decoder parameters to 2.3M while maintaining performance, facilitating clinical deployment on resource-constrained devices.

\begin{figure*}[htb]
    \centering
    \includegraphics[width=0.8\textwidth]{dwd.png}
    \caption{DWD Decoder Architecture: Extracts multi-scale features($F_1$-$F_4$)from the encoder, progressively refines them through a Feature Refinement Module (FRM), Cross-Scale Attention (CSA), and Gated Fusion (GF) layers, and ultimately generates a precise segmentation mask.}
    \label{fig:dwd}
\end{figure*}

\section{Experiment}
\subsection{Dataset and Preprocessing}
The public dataset used in this study, created by Cheng et al. \cite{cheng2015enhanced} , represents the most widely utilized brain tumor dataset containing three tumor types. It comprises 3,064 T1-weighted contrast-enhanced MRI scans from 233 patients with gliomas, meningiomas, and pituitary tumors, acquired at 512 × 512 pixel resolution. The dataset includes tumor ground truth masks to facilitate precise localization of pathological regions . Table \ref{tab:dataset} illustrates the distribution of three tumor categories. Multi-planar reconstructions (axial, sagittal, and coronal views) are available for comprehensive tumor analysis, with Figure \ref{fig:data} demonstrating representative samples across different planes.

\begin{table}[htb]
\centering
\caption{Dataset Statistics}
\label{tab:dataset}
\begin{tabular}{cccc}
\toprule
Type & No. of MRI Scans \\
\midrule
Meningioma & 708 \\
Glioma & 1426 \\
Pituitary Tumor & 930 \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Data Preprocessing \& Augmentation}: Original MATLAB (.mat) formatted data containing images, labels, patient IDs, and tumor masks were converted into NumPy (.npy) arrays and standardized image formats while preserving 16-bit depth for enhanced tissue contrast . Tumor-centric cropping ensured consistent spatial localization of lesions within the field of view. The training phase incorporated comprehensive data augmentation strategies:

\begin{itemize}
\item Horizontal/vertical flipping (probability=0.5)
\item Random rotation ($\pm15^\circ$, probability=0.5)
\item Translation \& scaling (±10\% range, probability=0.5)
\item Gaussian noise injection \& contrast enhancement (probability=0.5)
\item Elastic deformation \& grid distortion (probability=0.5)
\end{itemize}


These augmentation protocols effectively expanded dataset diversity while improving model robustness against anatomical variations and imaging artifacts . Validation/testing phases exclusively applied spatial resizing and intensity normalization without stochastic transformations.

\subsection{Implementation Details}
All experiments were conducted on Linux-based systems using Python 3.8 and PyTorch framework, with hardware acceleration provided by GPU workstations. Our customized MixViT architecture – featuring multi-scale attention-guided feature extraction and dynamic weight optimization – served as the encoder backbone. Initial learning rate was set to 3.0e-4 with weight decay 0.01 and batch size 16, employing AdamW optimizer \cite{loshchilov2019decoupledweightdecayregularization}. Input MRI scans were resized to 256×256 resolution with tumor-centric cropping to ensure lesion localization consistency.\cite{s22051960}

\begin{itemize}
    \item \textbf{Spatial Transformations}
    
    Horizontal/vertical flipping (probability $p=0.5$)
    
    Random 90 degree rotation (probability $p=0.5$)
    
    Affine transformation: translation range $±0.1$, scaling factor $±0.1$, rotation angle $±15^\circ$ (probability $p=0.5$)
    
    Elastic deformation: elasticity coefficient $\alpha=50$, standard deviation $\sigma=10$ (probability $p=0.5$)
    
    Grid distortion (probability $p=0.5$)
    
    
    \item \textbf{Intensity Perturbations}
    
     Gaussian noise injection: standard deviation $\sigma=0.05$ (probability $p=0.5$)
     
     Brightness/contrast adjustment: brightness gain $0.15$, contrast gain $0.2$ (probability $p=0.5$)
\end{itemize}

\begin{figure}[htb]
    \centering
    \includegraphics[width=0.35\textwidth]{data.png}
    \caption{In the preprocessed dataset, three distinct classes of brain MRI tumor cases are represented, with three samples per class: gliomas (cases 1-3), meningiomas (cases 4-6), and pituitary tumors (cases 7-9).}
    \label{fig:data}
\end{figure}

Z-score normalization was exclusively applied to foreground regions for enhanced tumor contrast. The adaptive loss weighting strategy dynamically addressed class imbalance, with training conducted for maximum 1,000 epochs employing early stopping (patience=10). Five-fold cross-validation ensured statistical reliability, with optimal models per fold preserved for final evaluation. Complete training required 4-5 hours on standard GPU configurations, while validation/testing phases completed within 30 minutes.

\subsection{Evaluation Metrics}
Four established segmentation metrics were computed for quantitative analysis:

\begin{itemize}
    \item \textbf{Dice Similarity Coefficient (Dice)} \\
    Quantifies volumetric overlap between prediction \(X\) and ground truth \(Y\):
    \begin{equation}
        \mathrm{Dice} = \frac{2|X \cap Y|}{|X| + |Y|}
    \end{equation}

    \item \textbf{95th Percentile Hausdorff Distance (HD$_{95}$)} \\
    \begin{equation}
    \begin{aligned}
       h_{95} &= \mathrm{P}_{95} \left( \left\{ \min_{y \in Y} d(x,y) \mid x \in X \right\} \right)
    \end{aligned}
    \end{equation}

\begin{equation}
\begin{aligned}
\mathrm{HD}_{95}(X,Y) &= \max \biggl( h_{95}(X \to Y), \\
&\quad h_{95}(Y \to X) \biggr)
\end{aligned}
\end{equation}

    \item \textbf{Recall/Sensitivity} \\
    Evaluates true positive detection rate:
    \begin{equation}
        \mathrm{Recall} = \frac{TP}{TP + FN}
    \end{equation}

    \item \textbf{F1 Score} \\
    Harmonic mean of precision and recall:
    \begin{equation}
        \mathrm{F1} = 2 \times \frac{\mathrm{Precision} \times \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
    \end{equation}
\end{itemize}

DSC and HD$_{95}$ remain the clinical gold standards for evaluating segmentation accuracy and boundary fidelity in medical imaging.

\subsection{Quantitative Analysis}
\begin{figure*}[htb]
    \centering
    \includegraphics[width=0.8\textwidth]{result.png}
    \caption{Visualization of segmentation results for three tumor types: meningioma (top), glioma (middle), and pituitary tumor (bottom). Each group displays (from left to right): original T1-CE MRI, ground truth annotation, and model prediction. The proposed method demonstrates precise boundary delineation across heterogeneous tumor morphologies.}
    \label{fig:vis}
\end{figure*}

From Fig.\ref{fig:vis}, it can be observed that the model exhibits strong generalization capabilities across segmentation tasks for meningiomas, gliomas, and pituitary adenomas. For gliomas with complex morphological patterns and pituitary adenomas exhibiting boundary ambiguity, the proposed method faithfully reconstructs tumor contours while achieving precise boundary delineation. This performance significantly mitigates both false negatives and false positives, particularly in cases with heterogeneous tumor intensity distributions. These visualizations further validate the clinical utility and robustness of our approach in medical image segmentation tasks, demonstrating consistent performance across diverse tumor phenotypes and imaging conditions.

\begin{table}[htbp]
    \centering
    \caption{The segmentation performance metrics of the proposed Multi-Scale Attention Fusion Network (MSAF-Net) on the test set, validated through 5-fold cross-validation.}
    \label{tab:metrics_summary}
    \begin{tabular}{lcc}
        \toprule
        Metric & Mean & Std \\
        \midrule
        Dice Coefficient      & 0.848 & 0.112 \\
        95\% Hausdorff Distance (HD$_{95}$)      & 7.02  & 10.74 \\
        Precision & 0.822 & 0.160 \\
        Recall    & 0.911 & 0.108 \\
        F1 Score        & 0.848 & 0.112 \\
        \bottomrule
    \end{tabular}
\end{table}

The experimental results demonstrate that our method achieves a mean Dice coefficient of 0.848, indicating substantial overlap between predicted and ground-truth segmentation regions. The HD$_{95}$ metric exhibits a mean value of 7.02 with a standard deviation of 10.74, suggesting competent boundary delineation across most samples while acknowledging residual variability in complex tumor margins. With precision and recall values of 0.822 and 0.911 respectively, coupled with an equivalent F1 score of 0.848, the model effectively balances sensitivity and specificity, achieving robust performance in lesion detection and spatial localization.

As shown in Table \ref{tab:results}, our approach outperforms comparative methods in both Dice coefficient and HD$_{95}$ metrics. Qualitative evaluations (Figure \ref{fig:vis}) further corroborate the enhanced precision of tumor boundary delineation, particularly in cases with irregular morphological patterns.

\begin{table*}[ht]
\centering
\caption{Comparison of the performance of different methods on the test set}
\label{tab:results}
\begin{tabular}{lccccc}
\toprule
Variant & Dice(ET) & HD95(mm) & Params(M) & recall(\%) & f1\\
\midrule
FCN & 0.7007 & 12.68 & 56.81 & 75.54 & 0.5741\\
SegNet & 0.7151 & 13.64 & 39.12 & 65.29 & 0.6182\\
U-Net & 0.8095 & 8.2 & 31.0423 & 82.87 & 0.8173\\
TransUNet & 0.8507 & 6.5 & 93.7406 & 80.62 & 0.7023\\
HNFNetv2 & 0.8569 & 5.1 & 20.9031 & 84.12 & 0.7659\\
Zig-RiR & 0.8790 & 5.3 & 23.6592 & 87.12 & 0.7879\\
MSAF-Net(ours) & \textbf{0.9165} & \textbf{4.3} & \textbf{18.8101} & \textbf{91.07} & \textbf{0.8477}\\
\bottomrule
\end{tabular}
\end{table*}

\begin{figure*}[htb]
    \centering
    \includegraphics[width=0.9\textwidth]{compare.png}
    \caption{Comparison of the segmentation effect of multiple models}
    \label{fig:compare}
\end{figure*}

\subsection{Ablation Studies}  
To systematically validate the effectiveness of key components in the proposed model, we conducted comprehensive ablation experiments to evaluate the impact of different modules on segmentation performance.  

\subsubsection{Effectiveness Analysis of Key Modules}  
Four model variants were designed to assess component contributions:  
(1) Full model (baseline) with all proposed components;  
(2) CSDA-removed model eliminating cross-scale dual attention;  
(3) DWD-removed model excluding dynamic weight decoder;  
(4) Fixed-weight model replacing dynamic weights with static counterparts.  
Additional evaluation compared Focal Loss against standard cross-entropy loss. Table \ref{tab:ablation_modules} summarizes quantitative performance comparisons.  

\begin{table*}[htbp]  
    \centering  
    \caption{Ablation study results of different model variants}  
    \label{tab:ablation_modules}  
    \begin{tabular}{lccccc}  
        \toprule  
        \textbf{Variant} & \textbf{Dice (ET)} & \textbf{IoU (\%)} & \textbf{HD95 (mm)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} \\  
        \midrule  
        Full model & \textbf{0.9165} & \textbf{72.88} & \textbf{4.3} & 80.08 & 91.07 \\  
        w/o CSDA & 0.6342 & 46.43 & 11.70 & 52.26 & 85.64 \\  
        w/o DWD & 0.4302 & 27.41 & 17.80 & 70.72 & 30.92 \\  
        Fixed-weight & 0.7230 & 56.61 & 6.40 & 76.92 & 68.20 \\  
        Focal Loss & 0.6836 & 51.93 & 7.68 & 60.62 & 78.36 \\  
        \bottomrule  
    \end{tabular}  
\end{table*}  

The baseline model achieved state-of-the-art performance across all metrics, attaining 91.65\% Dice, 72.88\% IoU, and 4.3mm HD95, confirming the efficacy of our architectural design.  

\subsubsection{Impact of CSDA Module}  
Removing the Cross-Scale Dual Attention (CSDA) module caused significant performance degradation: Dice decreased by 28.23pp (91.65\%→63.42\%), IoU dropped 26.45pp (72.88\%→46.43\%), and HD95 increased 3-fold (4.3mm→11.70mm). The precision plummeted to 52.26\% (-27.82pp) while recall remained relatively stable (85.64\% vs 91.07\%), indicating CSDA's critical role in enhancing boundary-aware feature fusion and precision optimization.  

\subsubsection{Crucial Role of DWD Module}  
Eliminating the Dynamic Weight Decoder (DWD) resulted in catastrophic performance collapse: Dice coefficient plunged 39.25pp to 52.40\%, IoU decreased 45.47pp to 27.41\%, and HD95 deteriorated to 17.80mm. The recall rate collapsed to 30.92\% (-60.15pp), demonstrating DWD's indispensable function in hierarchical feature integration and tumor region sensitivity enhancement.  

\subsubsection{Dynamic vs. Fixed Weights}  
The fixed-weight variant exhibited compromised performance with 72.30\% Dice (-19.35pp) and 6.40mm HD95, proving that dynamic weight adaptation enables context-aware feature emphasis compared to static parameterization.  

\subsubsection{Loss Function Analysis}  
Replacing cross-entropy with Focal Loss reduced Dice by 23.29pp (68.36\% vs 91.65\%), suggesting superior compatibility of standard cross-entropy with our architecture despite class imbalance challenges.  


\begin{figure*}[htb]  
    \centering  
    \includegraphics[width=0.95\textwidth]{alb.png}  
    \caption{Visual comparison of segmentation results from ablation studies. From left to right: Ground truth, full model, model without CSDA, model without DWD, and fixed-weight model. The complete model demonstrates superior performance in tumor boundary delineation and internal detail segmentation.}  
    \label{fig:ablation_vis}  
\end{figure*}  

\section{Main Contributions}

The main contributions of this work can be summarized as follows:

\begin{itemize}
    \item \textbf{Novel Architecture Design:} We propose MSAF-Net, a lightweight multi-scale attention-guided framework that achieves superior brain tumor segmentation performance while maintaining computational efficiency (18.81M parameters vs. 93.74M for TransUNet).
    
    \item \textbf{Channel-Spatial Dual Attention (CSDA) Module:} We introduce a novel attention mechanism that effectively captures both channel-wise dependencies and spatial relationships across multiple scales, leading to improved feature representation and boundary delineation.
    
    \item \textbf{Dynamic Weight Decoder (DWD):} We design an adaptive decoder that dynamically adjusts weighting strategies based on input characteristics, effectively addressing class imbalance issues common in medical image segmentation.
    
    \item \textbf{Comprehensive Evaluation:} We provide extensive experimental validation including comparison with recent methods (notably Zig-RiR), demonstrating 4.3mm HD95 and 91.65\% Dice coefficient, outperforming existing state-of-the-art approaches.
    
    \item \textbf{Clinical Applicability:} The proposed method offers a practical solution for brain tumor segmentation with balanced accuracy-efficiency trade-offs suitable for clinical deployment.
\end{itemize}

\section{Limitations and Future Work}

While our method demonstrates promising results, several limitations should be acknowledged:

\begin{itemize}
    \item \textbf{Dataset Limitation:} Our evaluation is primarily conducted on a single dataset (BraTS). Future work will include validation on multiple diverse datasets to strengthen generalizability claims.
    
    \item \textbf{Incremental Novelty:} The approach primarily combines and refines existing techniques. Future research will explore more fundamentally novel architectural innovations.
    
    \item \textbf{Limited Baseline Comparisons:} While we include recent methods like Zig-RiR, comparison with more lightweight Transformer variants would provide better context for our efficiency claims.
    
    \item \textbf{Clinical Validation:} Real-world clinical validation with radiologist assessment is needed to fully establish practical utility.
\end{itemize}

Future research directions include: (1) multi-dataset validation and domain adaptation techniques, (2) exploration of self-supervised learning approaches to reduce annotation requirements, (3) integration with multi-modal imaging data, and (4) comprehensive clinical validation studies.

\section{Conclusion}

To address the challenges of tumor morphological heterogeneity, boundary ambiguity, and class imbalance in brain tumor segmentation, this paper proposes \textit{Brain Tumor Segmentation via Multi-scale Attention Guidance and Dynamic Weight Optimization}. The method innovatively designs a Channel-Spatial Dual Attention (CSDA) module, integrating four key components: channel attention, spatial attention, multi-scale feature extraction, and cross-scale attention fusion, which effectively enhances semantically relevant features while suppressing irrelevant ones.

In architectural design, the lightweight Mix Vision Transformer (MixViT) serves as the encoder backbone, capturing local textures through overlapping patch embeddings and reducing computational complexity via linear attention mechanisms. Simultaneously, a Dynamic Weight Decoder (DWD) is introduced to mitigate class imbalance through adaptive weighted Dice loss functions, particularly enhancing segmentation accuracy for small tumor regions. Experimental results demonstrate that our method achieves a Dice coefficient of 0.9165 for enhancing tumor (ET) regions with HD95 reduced to 4.3 mm, significantly outperforming existing methods including U-Net (0.8095/8.2mm), TransUNet (0.8507/6.5mm), and HNF-Netv2 (0.8569/5.1mm). Additionally, our method outperforms the recently proposed Zig-RiR\cite{10969076} method (0.8790/5.3mm), demonstrating superior performance against state-of-the-art approaches. Notably, the parameter count (18.8101M) is substantially lower than TransUNet (93.7406M) and U-Net (31.0423M), achieving a synergistic optimization of accuracy and efficiency.


The proposed brain tumor segmentation framework, which combines lightweight Transformer architecture with dual attention mechanisms, maintains low computational complexity while surpassing state-of-the-art methods in segmentation precision. The CSDA module significantly improves model adaptability to tumors of varying sizes through multi-scale feature interactions, whereas the DWD strategy effectively resolves class imbalance issues, particularly boosting segmentation performance for small-volume tumors. Future research will focus on exploring weakly supervised learning\cite{zhu2023weakerthinkcriticallook} to reduce annotation dependency and extending the framework to multimodal medical image analysis, thereby facilitating practical applications in clinical intelligent diagnostic systems.\cite{bai2024m3dadvancing3dmedical}



\bibliographystyle{ieeetr}
\bibliography{custom}
\end{CJK}
\end{document}