# Awesome One-Step Generation

![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)  
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

*"Nature is pleased with simplicity."*  
— Isaac Newton

A curated list of papers, code, and resources for **one-step diffusion models** — methods that turn noise into high-quality samples in a *single* neural network forward pass.

---

## 🧭 Classification Rationale

Instead of the usual *native vs. distillation* split, we group methods by **the core mathematical principle that collapses iterative sampling into one step**. Each paper is placed by its *primary contribution*, with cross-references for works that span multiple principles.

| Legend         | Meaning                                                                      |
| -------------- | ---------------------------------------------------------------------------- |
| 🟢 **Native**  | Trained from scratch / teacher-free; one-step is a native training objective |
| 🔵 **Distill** | Distills a pretrained multi-step diffusion teacher into one/few steps        |
| 🟡 **Hybrid**  | Supports both native training and distillation                               |

> [!NOTE]
> 从[原github项目](https://github.com/Luciferbobo/Awesome-One-Step-Generation)复制来的框架，后文中所有蓝色note栏是我写的理解和笔记
> **1. Consistency Models (一致性模型)**
> - **工程机制：** 在模型训练中引入一致性损失，强行约束模型在同一条去噪常微分方程 (ODE) 轨迹上，能够从任意时间节点直接计算出相同的终点图像。
> - **数学本质：** 对于轨迹上的任意两点 x_t和 x_{t'}（两个时间点是相邻的），网络映射必须相同。
> - **代表技术：** LCM, LCM-LoRA, CTM。
> - **核心特征：** 兼容性极强，可通过 LoRA 形式作为独立加速模块插入现有模型，但容易损失高频细节。（为何兼容？）  
> **2. Rectified Flow(轨迹拉直)**
> - **工程机制：** 在扩散的前向过程（加噪阶段）进行根本性修改。不再使用复杂的高斯分布演变，而是通过线性插值，将纯噪声到真实数据的概率流轨迹定义为直线。
> - **数学本质：** 构造一条恒定速度的向量场 (Constant Velocity Vector Field)，因为积分路径变为直线，求解时截断误差大幅减小，从而允许极大的步长。
> - **代表技术：** Rectified Flow, InstaFlow, Stable Diffusion 3, Flux。
> - **核心特征：** 属于底座模型级别的架构革新，原生支持少步数，细节保留度极高。  
> **3. Distribution Divergence Minimization (分布散度最小化 / 蒸馏)**
> - **工程机制：** 将“教师模型（多步网络）的输出分布”和“学生模型（单步网络）的输出分布”视为两个高维概率空间中的概率密度函数，直接在整个分布层面上优化它们之间的距离。
> - **数学本质：** 通过优化 KL 散度、变分上界，或引入对抗网络（GAN 判别器本质是优化 JS 散度或 Wasserstein 距离），迫使单步生成器的输出概率密度逼近目标概率密度。
> - **代表技术：** DMD, SDXL-Lightning, ADD。
> - **核心特征：** 结合对抗训练后，画质上限极高，但对 CFG Scale 等推理参数高度敏感。  
> 以下三个更不熟悉  
> **4. 确定性高阶数值求解器 (Advanced ODE/SDE Solvers)**
> - **工程机制：** 不改变扩散模型本身的权重和训练方式，纯粹依靠数学优化推理阶段的微分方程求解算法，在保证误差可控的前提下，直接跳过多余的计算节点。
> - **数学本质：** 采用二阶、三阶的泰勒展开或指数积分方法（Exponential Integrator）来求解扩散 ODE，而非基础的欧拉方法。
> - **代表技术：** DDIM, DPM-Solver, Euler A。
> - **定位：** 这是将传统 1000 步压缩到 10-20 步的基石技术。  
> **5. 捷径与直接映射模型 (Shortcut / Direct Mapping Models)**
> - **工程机制：** 完全放弃马尔可夫链和“逐步去噪”的数学假设，通过极高容量的神经网络，直接学习从噪声空间到图像空间的非线性映射矩阵。
> - **数学本质：** 将常微分方程的求解过程替换为单次前向传递 (Feed-forward)，通常需要结合感知损失 (Perceptual Loss) 和结构重建损失。
> - **代表技术：** D2O (Diffusion to ODE), 部分基于 VAE/GAN 变体的直接生成网络。  
> **6. 漂移模型 (Drifting Models)**
> - **工程机制：** 在扩散过程的随机微分方程 (SDE) 中，通过修改控制粒子运动趋势的“漂移项”，引入指向目标数据分布的强引导向量。
> - **数学本质：** 修改原始 SDE 表达式 dx = f(x,t)dt + g(t)dw，在反向推理时加入基于数据特征的梯度引导，缩短收敛所需的积分时间。
> - **代表技术：** Generative Drifting, Gradient Flow Drifting。

---

## Table of Contents

- [Consistency Models](#consistency-models) — *Consistency Models, iCT, LCM, CTM*
- [Flow Straightening](#flow-straightening) — *Rectified Flow, Flow Matching, MeanFlow, InstaFlow*
- [Distribution Divergence Minimization](#distribution-divergence-minimization) — *DMD, SiD, Diff-Instruct, SwiftBrush*
  - [Forward KL / Regression](#forward-kl--regression)
  - [Reverse KL / Distribution Matching](#reverse-kl--distribution-matching)
  - [Fisher Divergence / Score Matching](#fisher-divergence--score-matching)
  - [Variational Bounds](#variational-bounds)
- [Adversarial Training](#adversarial-training) — *ADD, LADD, YOSO, SDXL-Lightning*
- [Shortcut / Direct Mapping](#shortcut--direct-mapping) — *Shortcut Models, D2O, NCT*
- [Drifting Models](#drifting-models) — *Generative Drifting, Gradient Flow Drifting*
- [Applications](#applications)
  - [Text-to-Image / Personalization](#text-to-image--personalization)
  - [Image Super-Resolution](#image-super-resolution)
  - [Video Generation](#video-generation)
  - [3D Generation](#3d-generation)
  - [Robotics / Control](#robotics--control)
  - [Other Applications](#other-applications)
- [Unified Frameworks & Theory](#unified-frameworks--theory)
- [Full Paper List](#full-paper-list-newest--oldest)
- [Related Resources](#related-resources)
- [Contributing](#contributing)

---

## Consistency Models

> **Core principle — self-consistency along the PF-ODE.** Any two points on the same probability-flow ODE trajectory must map to the *same* endpoint. Enforcing this consistency constraint lets a network jump directly from noise to data. Following iCT, the principle admits both consistency *training* (CT, 🟢 native) and consistency *distillation* (CD, 🔵 teacher-based).

### Native (🟢)

- **Consistency Models** [ICML 2023] 🟡  
[[Paper](https://arxiv.org/abs/2303.01469)]  
Self-consistency for direct noise-to-data mapping; supports both distillation (CD) and native training (CT).

> [!NOTE]
> 首先，在同一条加噪轨迹上找相邻的两个点（当前步和稍微清晰一点的下一步）,把他们输入到同一个模型里。通过 Loss Function 强制要求：这两个点预测出的“清晰原图”必须完全一模一样。这就是模型最根本的学习目标。  
> 但是这个目标产生了一个问题：如果模型产生纯黑的图（一样但是不相干的图）， loss为0。  
> 为了解决这个问题（提出了EMA,Exponential Moving Average机制），才把预测 T=T0 这一步的模型和预测 T=T0+1 这一步的模型，换成了两个不同的模型，一个叫学生模型，一个叫 目标模型。那么，目标模型每次只吸收学生模型一小部分梯度变化。  
> 这篇论文提出的方法既可以 from scratch 训练一个新的模型，也可以用于对传统模型的蒸馏，分别称作一致性训练和一致性蒸馏：
> 1. 一致性训练：目标模型和学生模型都是随机初始化的。
> 2. 一致性蒸馏：这两个模型的参数 θ 都是从老师模型里继承的。

- **Improved Techniques for Training Consistency Models (iCT)** [ICLR 2024] 🟢  
[[Paper](https://arxiv.org/abs/2310.14189)]  
Makes consistency training from scratch competitive with distillation.
- **Consistency Models Made Easy** [ICLR 2025] 🟢  
[[Paper](https://arxiv.org/abs/2406.14548)]  
Simplified and efficient consistency-model training recipe.
- **Inverse Flow and Consistency Models** [ICML 2025] 🟢；  
[[Paper](https://arxiv.org/abs/2502.11333)]  
Inverse-flow approach for consistency model training.
- **How to Build a Consistency Model: Learning Flow Maps via Self-Distillation** [NeurIPS 2025] 🟡  
[[Paper](https://arxiv.org/abs/2505.18825)] [[Code](https://github.com/nmboffi/flow-maps)]  
A systematic approach to learning flow maps via self-distillation.

### Distillation (🔵)

- **Latent Consistency Models (LCM)** [ICLR 2024] 🔵  
[[Paper](https://arxiv.org/abs/2310.04378)]  
Consistency distillation in latent space from pretrained Stable Diffusion.
- **Consistency Trajectory Models (CTM)** [ICLR 2024] 🔵  
[[Paper](https://arxiv.org/abs/2310.02279)] [[Code](https://github.com/sony/ctm)]  
Learning the probability-flow ODE trajectory for consistency.
- **SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation** [ICCV 2025] 🔵  
[[Paper](https://arxiv.org/abs/2503.09641)]  
Continuous-time consistency distillation for ultra-fast text-to-image.
- **Simple Distillation for One-Step Diffusion Models (CED)** [NeurIPS 2025] 🔵  
[[Paper](https://openreview.net/forum?id=NHw8muIAcL)]  
Contrastive energy distillation for one-step diffusion.
- **Phased Consistency Models (PCM)** [NeurIPS 2024] 🔵  
[[Paper](https://arxiv.org/abs/2405.18407)] [[Code](https://github.com/G-U-N/Phased-Consistency-Model)]  
Phased consistency distillation that surpasses LCM; supports 1-16 step generation.
- **Hyper-SD: Trajectory Segmented Consistency Model** [NeurIPS 2024] 🔵  
[[Paper](https://arxiv.org/abs/2404.13686)] [[Code](https://huggingface.co/ByteDance/Hyper-SD)]  
Trajectory-segmented consistency distillation with human feedback; surpasses SDXL-Lightning.
- **Truncated Consistency Models** [2024] 🟢  
[[Paper](https://openreview.net/forum?id=ZYDEJEvCbv)]  
Operating on truncated input time ranges for improved one-step and two-step quality.
- **Any-step Generation via N-th Order Recursive Consistent Velocity Field (RCGM)** [ICLR 2026] 🟢  
[[Paper](https://openreview.net/forum?id=RCGM)] [[Code](https://github.com/LINs-lab/RCGM)]  
N-th order recursive consistent velocity field estimation supporting any-step generation.

---

## Flow Straightening

> **Core principle — straighten the transport path.** If the trajectory connecting noise and data is (close to) a straight line, a single Euler step approximates the full ODE integration. Rectified-flow and flow-matching methods learn or refine such straight paths so that one-step Euler ≈ the complete ODE solution.

### Native (🟢)

> **Rectified Flow** and **Flow Matching** are catalogued under [Related Resources → Foundational Works](#foundational-works) as general frameworks. The methods below specifically target or achieve one-step generation.

- **Optimal Flow Matching: Learning Straight Trajectories in Just One Step** [NeurIPS 2024] 🟢  
[[Paper](https://arxiv.org/abs/2403.13117)] [[Code](https://github.com/Jhomanik/Optimal-Flow-Matching)]  
Restricting flow matching to obtain one-step straight trajectories.
- **Improving the Training of Rectified Flows** [NeurIPS 2024] 🟢  
[[Paper](https://arxiv.org/abs/2405.20320)] [[Code](https://github.com/sangyun884/rfpp)]  
Improved rectified-flow training enabling stronger one-step performance.
- **End-to-End Single-Step Flow Matching via Direct Models (FlowFit)** [ICLR 2025] 🟢  
[[Paper](https://openreview.net/forum?id=XpOnko2c59)]  
Direct model learning for single-step flow matching.
- **Mean Flows for One-step Generative Modeling (MeanFlow)** [NeurIPS 2025] 🟢  
[[Paper](https://arxiv.org/abs/2505.13447)] [[Code](https://github.com/haidog-yaqub/MeanFlow)]  
Native one-step generation via mean-velocity-field properties; trains from scratch.
- **One-step Latent-free Image Generation with Pixel Mean Flows (pMF)** [2026] 🟢  
[[Paper](https://arxiv.org/abs/2601.22158)]  
Latent-free one-step generation in pixel space; decouples output space from loss space, reaching 2.22 FID on ImageNet 256×256 and 2.48 FID on 512×512.
- **SoFlow: Solution Flow Models for One-Step Generative Modeling** [ICLR 2026] 🟢  
[[Paper](https://arxiv.org/abs/2512.15657)] [[Code](https://github.com/zlab-princeton/SoFlow)]  
Learning the solution map of flow ODEs for direct one-step generation.
- **Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow** [ICLR 2025] 🟢  
[[Paper](https://openreview.net/forum?id=rectified-diffusion)] [[Code](https://github.com/G-U-N/Rectified-Diffusion)]  
Challenges the straightness assumption; proposes a broader rectified flow framework.
- **Variational Flow Matching (S-VFM)** [CVPR 2026] 🟢  
[[Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Ma_Learning_Straight_Flows_Variational_Flow_Matching_for_Efficient_Generation_CVPR_2026_paper.pdf)]  
Learning straight flows via variational latent codes integrated with the flow-matching objective.
- **Beyond Optimal Transport: Model-Aligned Coupling for Flow Matching** [CVPR 2026 Findings] 🟢  
[[Paper](https://openaccess.thecvf.com/content/CVPR2026F/html/Lin_Beyond_Optimal_Transport_Model-Aligned_Coupling_for_Flow_Matching_CVPRF_2026_paper.html)] [[Code](https://github.com/tmllab/2026_CVPR_MAC)]  
Model-aligned coupling selects learnable source-target pairs to improve one-step/few-step flow matching generation.
- **SubFlow: Sub-mode Conditioned Flow Matching for Diverse One-Step Generation** [arXiv 2026] 🟢  
[[Paper](https://arxiv.org/abs/2604.12273)]  
Sub-mode conditioning reduces averaging distortion and improves diversity in one-step flow matching.

### Distillation (🔵)

> **Progressive Distillation** is catalogued under [Related Resources → Foundational Works](#foundational-works) as it is a general step-compression method rather than a one-step-specific technique. It is cross-referenced here as it inspired much of the flow-distillation lineage.

- **Bi-Anchor Interpolation Solver for Accelerating Generative Modeling** [ICML 2026] 🔵  
[[Paper](https://arxiv.org/abs/2601.21542)] [[Code](https://github.com/HKUST-LongGroup/BA-solver)]  
Lightweight training-based solver for flow matching models.  
- **InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation** [ICLR 2024] 🔵  
[[Paper](https://arxiv.org/abs/2309.06380)] [[Code](https://github.com/gnobitab/InstaFlow)]  
Rectified-Flow reflow technique for one-step text-to-image.
- **Training Smaller One-Step Diffusion Models with Rectified Flow (SlimFlow)** [ECCV 2024] 🔵  
[[Paper](https://arxiv.org/abs/2407.12718)] [[Code](https://github.com/yuanzhi-zhu/SlimFlow)]  
Parameter-efficient one-step models via rectified-flow compression.
- **Learning Few-Step Diffusion Models by Trajectory Distribution Matching (TDM)** [ICCV 2025] 🔵  
[[Paper](https://arxiv.org/abs/2503.06674)] [[Code](https://github.com/Luo-Yihong/TDM)]  
Unified trajectory and distribution matching for few-step distillation.
- **Align Your Flow: Scaling Continuous-Time Flow Map Distillation** [NeurIPS 2025] 🔵  
[[Paper](https://arxiv.org/abs/2506.14603)]  
Continuous-time objectives for training flow maps at scale.
- **TRACT: Denoising Diffusion Models with Transitive Closure Time-Distillation** [2023] 🔵  
[[Paper](https://arxiv.org/abs/2303.04248)] [[Code](https://github.com/apple/ml-tract)]  
Transitive closure time-distillation for progressive step reduction.
- **Piecewise Rectified Flow (PeRFlow)** [NeurIPS 2024] 🔵  
[[Paper](https://arxiv.org/abs/2405.07510)] [[Code](https://github.com/magic-research/piecewise-rectified-flow)]  
Universal plug-and-play accelerator via piecewise linearization of rectified flows.

---

## Distribution Divergence Minimization

> **Core principle — minimize a statistical divergence between distributions.** These methods train the one-step generator so that its output distribution `p_θ` matches the target distribution `p_target` by minimizing some divergence `D(p_θ ∥ p_target)`. As shown by **Uni-Instruct** (NeurIPS 2025), DMD, SiD, Diff-Instruct and many others are special cases of a single *diffusion divergence minimization* framework; they differ chiefly in **which divergence** they minimize. We organize them along that axis.

### Forward KL / Regression

> Minimizing the forward KL `D(p_target ∥ p_θ)` is mode-covering and reduces to maximum-likelihood / regression-style objectives.

- **Progressive Distillation** is catalogued under [Related Resources → Foundational Works](#foundational-works), since its core contribution is *step compression* rather than a divergence-minimization result. It is cross-referenced here because its regression-on-teacher objective is forward-KL in spirit.
- **BOOT: Data-free Distillation of Denoising Diffusion Models with Bootstrapping** [ICML 2023] 🔵  
[[Paper](https://arxiv.org/abs/2306.05544)]  
Data-free bootstrapping distillation via sequential prediction on the signal ODE.
- **EM Distillation for One-step Diffusion Models** [NeurIPS 2024] 🔵  
[[Paper](https://arxiv.org/abs/2405.16852)]  
Maximum-likelihood (forward-KL) distillation via an Expectation-Maximization formulation.

### Reverse KL / Distribution Matching

> Minimizing the reverse KL `D(p_θ ∥ p_target)` is mode-seeking; in practice it is estimated through the score functions of the real and the generator ("fake") distributions.

- **One-step Diffusion with Distribution Matching Distillation (DMD)** [CVPR 2024] 🔵  
[[Paper](https://arxiv.org/abs/2311.18828)]  
Learning score functions of real and fake distributions for reverse-KL distribution matching.
- **Improved Distribution Matching Distillation (DMD2)** [NeurIPS 2024 Oral] 🔵  
[[Paper](https://arxiv.org/abs/2405.14867)] [[Code](https://github.com/tianweiy/dmd2)]  
Improved DMD that removes the costly regression loss.
- **One-step Diffusion Models with f-Divergence Distribution Matching (f-distill)** [2025] 🔵  
[[Paper](https://arxiv.org/abs/2502.15681)]  
A generalized f-divergence framework subsuming reverse-KL distribution matching.

### Fisher Divergence / Score Matching

> Matching the *scores* (gradients of log-density) of the generator and target corresponds to minimizing a Fisher-type divergence — the basis of score-identity and score-implicit methods.

- **Score identity Distillation (SiD)** [ICML 2024] 🔵  
[[Paper](https://arxiv.org/abs/2404.04057)] [[Code](https://github.com/mingyuanzhou/SiD)]  
Data-free score-identity matching for exponentially fast distillation.
- **One-Step Diffusion Distillation through Score Implicit Matching (SIM)** [NeurIPS 2024] 🔵  
[[Paper](https://arxiv.org/abs/2410.16794)] [[Code](https://github.com/maple-research-lab/SIM)]  
Score implicit matching that preserves the teacher's denoising capability.
- **DiffRatio: Training One-Step Diffusion Models Without Teacher Supervision** [ICLR 2025] 🟢  
[[Paper](https://arxiv.org/abs/2502.08005)] [[Code](https://github.com/Wenlin-Chen/DiffRatio)]  
Direct score-ratio estimation for teacher-free one-step training.
- **Adversarial Score identity Distillation (SiDA)** [ICLR 2025] 🔵  
[[Paper](https://arxiv.org/abs/2410.14919)] [[Code](https://github.com/mingyuanzhou/SiD/tree/sida)]  
Adversarial enhancement of SiD, surpassing the teacher model.
- **Score-of-Mixture Training** [ICML 2025 Spotlight] 🟢  
[[Paper](https://arxiv.org/abs/2502.09609)]  
Minimizing mixture-distribution divergence for pretrain-free one-step generation.

### Variational Bounds

> When the divergence cannot be optimized directly, a tractable *variational bound* (e.g. variational score distillation) is minimized instead, using a learned score/critic to guide the generator.

- **Diff-Instruct: A Universal Approach for Transferring Knowledge from Pre-trained Diffusion Models** [NeurIPS 2023] 🔵  
[[Paper](https://arxiv.org/abs/2305.18455)]  
Universal framework using variational bounds to guide any generator; conceptual precursor to many one-step methods.
- **SwiftBrush: One-Step Text-to-Image Diffusion Model with Variational Score Distillation** [CVPR 2024] 🔵  
[[Paper](https://arxiv.org/abs/2312.05239)] [[Code](https://github.com/VinAIResearch/SwiftBrush)]  
Image-free variational score distillation for one-step text-to-image.
- **SwiftBrush v2: Make Your One-step Model Better Than Its Teacher** [ECCV 2024] 🔵  
[[Paper](https://arxiv.org/abs/2408.14176)] [[Code](https://github.com/vinairesearch/swiftbrushv2)]  
Improved SwiftBrush that surpasses teacher-model performance.
- **Diff-Instruct++: Training One-step Text-to-image Generator Model to Human-Preferred** [2024] 🔵  
[[Paper](https://arxiv.org/abs/2410.18881)] [[Code](https://github.com/pkulwj1994/diff_instruct_pp)]  
Human-preference alignment for one-step text-to-image generators.
- **WaDi: Weight Direction-aware Distillation for One-step Image Synthesis** [CVPR 2026] 🔵  
[[Paper](https://arxiv.org/abs/2603.08258)] [[Code](https://github.com/gudaochangsheng/WaDi)]  
Weight-direction-aware VSD framework with LoRaD adapters, improving one-step distillation with about 10% trainable parameters.

---

## Adversarial Training

> **Core principle — push samples onto the data manifold with a discriminator.** Borrowing the GAN principle, an adversarial discriminator drives generated samples toward the real-data distribution, yielding sharp, high-fidelity outputs in a single step. Often combined with score- or consistency-based objectives.

- **You Only Sample Once (YOSO)** [ICLR 2024] 🟢  
[[Paper](https://arxiv.org/abs/2403.12931)] [[Code](https://github.com/Luo-Yihong/YOSO)]  
Self-cooperative diffusion-GAN for one-step text-to-image, trained without a frozen teacher.
- **Adversarial Diffusion Distillation (ADD)** [ECCV 2024] 🔵  
[[Paper](https://arxiv.org/abs/2311.17042)]  
Adversarial training for distilling diffusion into 1–4 step generators (SDXL-Turbo).
- **Latent Adversarial Diffusion Distillation (LADD)** [ECCV 2024] 🔵  
[[Paper](https://arxiv.org/abs/2403.12015)]  
Latent-space adversarial distillation overcoming ADD limitations (powers FLUX.1 [schnell]).
- **SDXL-Lightning: Progressive Adversarial Diffusion Distillation** [2024] 🔵  
[[Paper](https://arxiv.org/abs/2402.13929)] [[Code](https://huggingface.co/ByteDance/SDXL-Lightning)]  
Combines progressive and adversarial distillation for one-step 1024px text-to-image on SDXL.
- **TwinFlow: Realizing One-step Generation with Self-adversarial Flows** [ICLR 2026] 🟢  
[[Paper](https://arxiv.org/abs/2512.05150)] [[Code](https://github.com/inclusionAI/TwinFlow)]  
Self-adversarial flow framework without fixed auxiliary models for one-step generation.

---

## Shortcut / Direct Mapping

> **Core principle — learn the noise-to-data map directly.** Rather than reasoning about trajectories or iterative denoising, these methods propose a new paradigm: a single conditioned network that maps noise straight to data, bypassing the ODE/SDE machinery entirely.

- **One Step Diffusion via Shortcut Models** [ICLR 2025 Oral] 🟢  
[[Paper](https://arxiv.org/abs/2410.12557)] [[Code](https://github.com/kvfrans/shortcut-models)]  
A single network and single training phase for direct generation without iterative denoising.
- **High-Order Matching for One-Step Shortcut Diffusion Models** [ICLR 2025] 🟢  
[[Paper](https://arxiv.org/abs/2502.00688)]  
High-order matching to improve shortcut models.
- **Revisiting Diffusion Models: From Generative Pre-training to One-Step Generation (D2O)** [ICML 2025] 🟢  
[[Paper](https://arxiv.org/abs/2506.09376)] [[Code](https://github.com/Zyriix/D2O)]  
Treating diffusion as generative pre-training and fine-tuning for one-step generation.
- **Noise Consistency Training (NCT)** [NeurIPS 2025] 🟢  
[[Paper](https://arxiv.org/abs/2506.19741)] [[Code](https://github.com/Luo-Yihong/NCT)]  
Native integration of new control signals directly into one-step generators.
- **Di[M]O: Distilling Masked Diffusion Models into One-step Generator** [ICCV 2025] 🔵  
[[Paper](https://arxiv.org/abs/2503.15457)] [[Code](https://github.com/yuanzhi-zhu/DiMO)]  
First work distilling discrete masked diffusion models into a one-step generator.

---

## Drifting Models

> **Core principle — evolve the generator's output distribution during training via Wasserstein gradient flow.** Instead of iterating at inference time, drifting models introduce a "drifting field" that continuously pushes the generator's output distribution toward the data distribution during training. At inference, a single forward pass suffices. Theoretically equivalent to minimizing KL divergence via Wasserstein-2 gradient flow and secretly performing score matching under Gaussian kernels.

- **Generative Modeling via Drifting** [2026] 🟢  
[[Paper](https://arxiv.org/abs/2602.04770)] [[Code](https://github.com/lambertae/drifting)]  
The foundational drifting model: evolves generator distribution at training time; achieves FID 1.54 on ImageNet 256×256 with one-step generation.
- **Generative Drifting is Secretly Score Matching: a Spectral and Variational Perspective** [2026] 🟢  
[[Paper](https://arxiv.org/abs/2603.09936)]  
Proves drifting under Gaussian kernels equals score difference of smoothed distributions; connects to JKO discretization.
- **Gradient Flow Drifting: Generative Modeling via Wasserstein Gradient Flows** [2026] 🟢  
[[Paper](https://arxiv.org/abs/2603.10592)]  
Establishes exact equivalence between drifting and Wasserstein gradient flow of KL divergence; extends to Riemannian manifolds.
- **W-Flow: One-Step Generative Modeling via Wasserstein Gradient Flows** [2026] 🟢  
[[Paper](https://arxiv.org/abs/2605.11755)] [[Code](https://github.com/hanjq17/W-Flow)]  
Trains a one-step generator with training dynamics guided by a Wasserstein gradient flow of the Sinkhorn divergence; achieves FID 1.29 on ImageNet 256×256 with one-step generation.

---

## Applications

### Text-to-Image / Personalization

- **PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion** [CVPR 2026] 🔵  
[[Paper](https://arxiv.org/abs/2602.12769)]  
Training-free patch-based low-step/one-step high-resolution text-to-image generation, targeting practical 4K synthesis.
- **Adversarial Concept Distillation for One-Step Diffusion Personalization (OPAD)** [CVPR 2026 Findings] 🔵  
[[Paper](https://arxiv.org/abs/2510.20512)]  
Teacher-student adversarial distillation for personalizing one-step text-to-image diffusion models.

### Image Super-Resolution

- **One-Step Effective Diffusion Network for Real-World Image Super-Resolution (OSEDiff)** [NeurIPS 2024] 🔵  
[[Paper](https://arxiv.org/abs/2406.08177)] [[Code](https://github.com/cswry/OSEDiff)]
- **One Step Diffusion-based Super-Resolution with Time-Aware Distillation (TAD-SR)** [NeurIPS 2024] 🔵  
[[Paper](https://arxiv.org/abs/2408.07476)] [[Code](https://github.com/LearningHx/TAD-SR)]
- **TSD-SR: One-Step Diffusion with Target Score Distillation** [CVPR 2025] 🔵  
[[Paper](https://arxiv.org/abs/2411.18263)] [[Code](https://github.com/Microtreei/TSD-SR)]
- **Consistency Trajectory Matching for One-Step Generative Super-Resolution** [ICCV 2025] 🔵  
[[Paper](https://arxiv.org/abs/2503.20349)] [[Code](https://github.com/LabShuHangGU/CTMSR)]
- **FlowSR: Fast Image Super-Resolution via Consistency Rectified Flow** [ICCV 2025] 🟢  
[[Paper](https://arxiv.org/abs/2605.12377)]
- **One-Step Diffusion Transformer for Controllable Real-World Image SR (ODTSR)** [CVPR 2026] 🔵  
[[Paper](https://arxiv.org/abs/2511.17138)] [[Code](https://github.com/RedMediaTech/ODTSR)]
- **TEASR: Training-efficient Any-step Diffusion Transformer for Real-World Image Super-Resolution** [2026] 🔵  
[[Paper](https://arxiv.org/abs/2606.16188)]  
Self-adversarial any-step Real-ISR distillation in a single diffusion model, supporting both one-step and multi-step restoration.

### Video Generation

- **SF-V: Single Forward Video Generation Model** [NeurIPS 2024] 🔵  
[[Paper](https://arxiv.org/abs/2406.04324)] [[Code](https://github.com/snap-research/SF-V)]
- **OSV: One Step is Enough for High-Quality Image to Video Generation** [CVPR 2025] 🔵  
[[Paper](https://arxiv.org/abs/2409.11367)]
- **Diffusion Adversarial Post-Training for One-Step Video Generation** [ICML 2025] 🔵  
[[Paper](https://arxiv.org/abs/2501.08316)]
- **DOVE: Efficient One-Step Diffusion for Real-World Video Super-Resolution** [NeurIPS 2025] 🔵  
[[Paper](https://arxiv.org/abs/2505.16239)] [[Code](https://github.com/zhengchen1999/DOVE)]
- **One-Step Diffusion for Detail-Rich Video Super-Resolution (DLoRAL)** [NeurIPS 2025] 🔵  
[[Paper](https://arxiv.org/abs/2506.15591)]
- **Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation** [NeurIPS 2025] 🔵  
[[Paper](https://arxiv.org/abs/2506.09350)]
- **Motion Consistency Model (MCM)** [NeurIPS 2024] 🔵  
[[Paper](https://arxiv.org/abs/2406.06890)] [[Code](https://github.com/yhZhai/mcm)]  
Single-phase video distillation with disentangled motion and appearance learning.
- **DCM: Dual-Expert Consistency Model for Video Generation** [ICCV 2025] 🔵  
[[Paper](https://arxiv.org/abs/2506.03123)]  
Parameter-efficient dual-expert consistency decoupling semantics and details.

### 3D Generation

- **DIFIX3D+: Improving 3D Reconstructions with Single-Step Diffusion Models** [CVPR 2025 Oral] 🔵  
[[Paper](https://arxiv.org/abs/2503.01774)] [[Code](https://github.com/nv-tlabs/Difix3D)]  
Single-step 3D reconstruction and novel view synthesis; CVPR 2025 Best Paper Finalist.
- **VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step** [CVPR 2025] 🔵  
[[Paper](https://arxiv.org/abs/2504.01956)] [[Code](https://github.com/THU-SI/VideoScene)]  
First one-step 3D scene generation by distilling video diffusion.

### Robotics / Control

- **OneDP: Fast Visuomotor Policies via Diffusion Distillation** [ICML 2025] 🔵  
[[Paper](https://arxiv.org/abs/2410.21257)]  
Distilling pretrained diffusion policies into single-step policies for robot control.

### Other Applications

- **OSDFace: One-Step Diffusion Model for Face Restoration** [CVPR 2025] 🔵  
[[Paper](https://arxiv.org/abs/2411.17163)] [[Code](https://github.com/jkwang28/OSDFace)]
- **Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models** [ICLR 2025] 🟢  
[[Paper](https://arxiv.org/abs/2409.08861)]
- **Reward Guided Latent Consistency Distillation (RG-LCD)** [ICLR 2025] 🔵  
[[Paper](https://arxiv.org/abs/2403.11027)]
- **Diffusion2GAN: Distilling Diffusion Models into Conditional GANs** [ECCV 2024] 🔵  
[[Paper](https://arxiv.org/abs/2405.05967)]
- **Flash Diffusion: Accelerating Any Conditional Diffusion Model for Few Steps Image Generation** [AAAI 2024] 🔵  
[[Paper](https://arxiv.org/abs/2406.02347)] [[Code](https://github.com/gojasper/flash-diffusion)]
- **DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture** [CVPR 2025] 🔵  
[[Paper](https://arxiv.org/abs/2409.03550)] [[Code](https://github.com/qianlong0502/DKDM)]

---

## Unified Theory

- **Uni-Instruct: One-step Diffusion Model through Unified Diffusion Divergence Instruction** [NeurIPS 2025]  
[[Paper](https://arxiv.org/abs/2505.20755)]  
A unified theoretical framework proving 10+ existing distillation methods are special cases of diffusion divergence minimization.
- **High-Order Flow Matching: Unified Framework** [NeurIPS 2025]  
[[Paper](https://openreview.net/forum?id=ib0aV2hphN)]  
A unified high-order flow-matching framework.
- **On the Design of One-step Diffusion via Shortcutting Flow Paths** [ICLR 2026] 🟢  
[[Paper](https://arxiv.org/abs/2512.11831)] [[Code](https://github.com/EDAPINENUT/ExplicitShortCut)]  
Theoretical analysis of supervision signal variance for designing one-step flow shortcuts.

---

## Full Paper List

> **Type:** 🟢 Native · 🔵 Distillation · 🟡 Hybrid  
> **Category:** CM = Consistency Models · FS = Flow Straightening · DDM = Distribution Divergence Min. · ADV = Adversarial Training · SC = Shortcut / Direct Mapping · DM = Drifting Models · APP = Applications · TH = Unified Theory · FW = Foundational Work

| Date    | Paper                                                                                          | Venue               | Type | Category | Links                                                                                                                                                                                                         |
| ------- | ---------------------------------------------------------------------------------------------- | ------------------- | ---- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026.06 | TEASR: Training-efficient Any-step Diffusion Transformer for Real-World Image Super-Resolution | 2026                | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2606.16188)]                                                                                                                                                                   |
| 2026.05 | W-Flow: One-Step Generative Modeling via Wasserstein Gradient Flows                            | 2026                | 🟢   | DM       | [[Paper](https://arxiv.org/abs/2605.11755)]                                                                                                                                                                   |
| 2026.05 | FlowSR: Fast Image SR via Consistency Rectified Flow                                           | ICCV 2025           | 🟢   | APP      | [[Paper](https://arxiv.org/abs/2605.12377)]                                                                                                                                                                   |
| 2026.04 | SubFlow: Sub-mode Conditioned Flow Matching for Diverse One-Step Generation                    | 2026                | 🟢   | FS       | [[Paper](https://arxiv.org/abs/2604.12273)]                                                                                                                                                                   |
| 2026.03 | Gradient Flow Drifting: Generative Modeling via Wasserstein Gradient Flows                     | 2026                | 🟢   | DM       | [[Paper](https://arxiv.org/abs/2603.10592)]                                                                                                                                                                   |
| 2026.03 | Generative Drifting is Secretly Score Matching: a Spectral and Variational Perspective         | 2026                | 🟢   | DM       | [[Paper](https://arxiv.org/abs/2603.09936)]                                                                                                                                                                   |
| 2026.03 | WaDi: Weight Direction-aware Distillation for One-step Image Synthesis                         | CVPR 2026           | 🔵   | DDM      | [[Paper](https://arxiv.org/abs/2603.08258)] [[Code](https://github.com/gudaochangsheng/WaDi)]                                                                                                                 |
| 2026.03 | SoFlow: Solution Flow Models for One-Step Generative Modeling                                  | ICLR 2026           | 🟢   | FS       | [[Paper](https://arxiv.org/abs/2512.15657)] [[Code](https://github.com/zlab-princeton/SoFlow)]                                                                                                                |
| 2026.02 | PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion   | CVPR 2026           | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2602.12769)]                                                                                                                                                                   |
| 2026.02 | Bi-Anchor Interpolation Solver for Accelerating Generative Modeling                            | ICML 2026           | 🔵   | SC       | [[Paper](https://arxiv.org/abs/2601.21542)] [[Code](https://github.com/HKUST-LongGroup/BA-solver)]                                                                                                            |
| 2026.02 | Generative Modeling via Drifting                                                               | 2026                | 🟢   | DM       | [[Paper](https://arxiv.org/abs/2602.04770)] [[Code](https://github.com/lambertae/drifting)]                                                                                                                   |
| 2026.01 | One-step Latent-free Image Generation with Pixel Mean Flows (pMF)                              | 2026                | 🟢   | FS       | [[Paper](https://arxiv.org/abs/2601.22158)]                                                                                                                                                                   |
| 2026.01 | DiffRatio: Training One-Step Diffusion Models Without Teacher Supervision                      | ICLR 2025           | 🟢   | DDM      | [[Paper](https://arxiv.org/abs/2502.08005)] [[Code](https://github.com/Wenlin-Chen/DiffRatio)]                                                                                                                |
| 2025.12 | High-Order Flow Matching: Unified Framework                                                    | NeurIPS 2025        | —    | TH       | [[Paper](https://openreview.net/forum?id=ib0aV2hphN)]                                                                                                                                                         |
| 2025.12 | TwinFlow: One-step Generation with Self-adversarial Flows                                      | ICLR 2026           | 🟢   | ADV      | [[Paper](https://arxiv.org/abs/2512.05150)] [[Code](https://github.com/inclusionAI/TwinFlow)]                                                                                                                 |
| 2025.12 | On the Design of One-step Diffusion via Shortcutting Flow Paths                                | ICLR 2026           | 🟢   | TH       | [[Paper](https://arxiv.org/abs/2512.11831)] [[Code](https://github.com/EDAPINENUT/ExplicitShortCut)]                                                                                                          |
| 2025.11 | ODTSR: One-Step Diffusion Transformer for Controllable Real-World Image SR                     | CVPR 2026           | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2511.17138)] [[Code](https://github.com/RedMediaTech/ODTSR)]                                                                                                                   |
| 2025.10 | Adversarial Concept Distillation for One-Step Diffusion Personalization (OPAD)                 | CVPR 2026 Findings  | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2510.20512)]                                                                                                                                                                   |
| 2025.11 | Consistency Trajectory Matching for One-Step Generative SR                                     | ICCV 2025           | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2503.20349)] [[Code](https://github.com/LabShuHangGU/CTMSR)]                                                                                                                   |
| 2025.11 | DOVE: Efficient One-Step Diffusion for Real-World Video SR                                     | NeurIPS 2025        | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2505.16239)] [[Code](https://github.com/zhengchen1999/DOVE)]                                                                                                                   |
| 2025.10 | One-Step Diffusion for Detail-Rich Video SR (DLoRAL)                                           | NeurIPS 2025        | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2506.15591)]                                                                                                                                                                   |
| 2025.10 | Autoregressive Adversarial Post-Training for Real-Time Interactive Video Gen                   | NeurIPS 2025        | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2506.09350)]                                                                                                                                                                   |
| 2025.10 | Uni-Instruct: One-step Diffusion via Unified Diffusion Divergence Instruction                  | NeurIPS 2025        | —    | TH       | [[Paper](https://arxiv.org/abs/2505.20755)]                                                                                                                                                                   |
| 2025.10 | How to Build a Consistency Model: Learning Flow Maps via Self-Distillation                     | NeurIPS 2025        | 🟡   | CM       | [[Paper](https://arxiv.org/abs/2505.18825)] [[Code](https://github.com/nmboffi/flow-maps)]                                                                                                                    |
| 2025.10 | Diffusion Adversarial Post-Training for One-Step Video Generation                              | ICML 2025           | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2501.08316)]                                                                                                                                                                   |
| 2025.10 | Any-step Generation via RCGM                                                                   | ICLR 2026           | 🟢   | CM       | [[Paper](https://openreview.net/forum?id=RCGM)] [[Code](https://github.com/LINs-lab/RCGM)]                                                                                                                    |
| 2025.09 | SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation                  | ICCV 2025           | 🔵   | CM       | [[Paper](https://arxiv.org/abs/2503.09641)]                                                                                                                                                                   |
| 2025.07 | Score-of-Mixture Training                                                                      | ICML 2025 Spotlight | 🟢   | DDM      | [[Paper](https://arxiv.org/abs/2502.09609)]                                                                                                                                                                   |
| 2025.06 | Noise Consistency Training (NCT)                                                               | NeurIPS 2025        | 🟢   | SC       | [[Paper](https://arxiv.org/abs/2506.19741)] [[Code](https://github.com/Luo-Yihong/NCT)]                                                                                                                       |
| 2025.06 | Align Your Flow: Scaling Continuous-Time Flow Map Distillation                                 | NeurIPS 2025        | 🔵   | FS       | [[Paper](https://arxiv.org/abs/2506.14603)]                                                                                                                                                                   |
| 2025.06 | Revisiting Diffusion Models: From Pre-training to One-Step Generation (D2O)                    | ICML 2025           | 🟢   | SC       | [[Paper](https://arxiv.org/abs/2506.09376)] [[Code](https://github.com/Zyriix/D2O)]                                                                                                                           |
| 2025.06 | Diff-Instruct++: Training One-step T2I Generator to Human-Preferred                            | 2024                | 🔵   | DDM      | [[Paper](https://arxiv.org/abs/2410.18881)] [[Code](https://github.com/pkulwj1994/diff_instruct_pp)]                                                                                                          |
| 2025.06 | One Step Diffusion via Shortcut Models                                                         | ICLR 2025 Oral      | 🟢   | SC       | [[Paper](https://arxiv.org/abs/2410.12557)] [[Code](https://github.com/kvfrans/shortcut-models)]                                                                                                              |
| 2025.06 | DCM: Dual-Expert Consistency Model for Video                                                   | ICCV 2025           | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2506.03123)]                                                                                                                                                                   |
| 2025.06 | Variational Flow Matching (S-VFM)                                                              | CVPR 2026           | 🟢   | FS       | [[Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Ma_Learning_Straight_Flows_Variational_Flow_Matching_for_Efficient_Generation_CVPR_2026_paper.pdf)]                                            |
| 2025.05 | TSD-SR: One-Step Diffusion with Target Score Distillation                                      | CVPR 2025           | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2411.18263)] [[Code](https://github.com/Microtreei/TSD-SR)]                                                                                                                    |
| 2025.05 | Beyond Optimal Transport: Model-Aligned Coupling for Flow Matching                             | CVPR 2026 Findings  | 🟢   | FS       | [[Paper](https://openaccess.thecvf.com/content/CVPR2026F/html/Lin_Beyond_Optimal_Transport_Model-Aligned_Coupling_for_Flow_Matching_CVPRF_2026_paper.html)] [[Code](https://github.com/tmllab/2026_CVPR_MAC)] |
| 2025.05 | Mean Flows for One-step Generative Modeling (MeanFlow)                                         | NeurIPS 2025        | 🟢   | FS       | [[Paper](https://arxiv.org/abs/2505.13447)] [[Code](https://github.com/haidog-yaqub/MeanFlow)]                                                                                                                |
| 2025.05 | VideoScene: One-Step 3D Scene Generation                                                       | CVPR 2025           | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2504.01956)] [[Code](https://github.com/THU-SI/VideoScene)]                                                                                                                    |
| 2025.04 | OSV: One Step is Enough for High-Quality Image to Video Generation                             | CVPR 2025           | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2409.11367)]                                                                                                                                                                   |
| 2025.03 | Learning Few-Step Diffusion Models by Trajectory Distribution Matching (TDM)                   | ICCV 2025           | 🔵   | FS       | [[Paper](https://arxiv.org/abs/2503.06674)] [[Code](https://github.com/Luo-Yihong/TDM)]                                                                                                                       |
| 2025.03 | DIFIX3D+: Single-Step 3D Reconstruction                                                        | CVPR 2025 Oral      | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2503.01774)] [[Code](https://github.com/nv-tlabs/Difix3D)]                                                                                                                     |
| 2025.02 | You Only Sample Once (YOSO)                                                                    | ICLR 2024           | 🟢   | ADV      | [[Paper](https://arxiv.org/abs/2403.12931)] [[Code](https://github.com/Luo-Yihong/YOSO)]                                                                                                                      |
| 2025.02 | One-step Diffusion Models with f-Divergence Distribution Matching (f-distill)                  | 2025                | 🔵   | DDM      | [[Paper](https://arxiv.org/abs/2502.15681)]                                                                                                                                                                   |
| 2025.02 | Inverse Flow and Consistency Models                                                            | ICML 2025           | 🟢   | CM       | [[Paper](https://arxiv.org/abs/2502.11333)]                                                                                                                                                                   |
| 2025.02 | High-Order Matching for One-Step Shortcut Diffusion Models                                     | ICLR 2025           | 🟢   | SC       | [[Paper](https://arxiv.org/abs/2502.00688)]                                                                                                                                                                   |
| 2025.02 | Simple Distillation for One-Step Diffusion Models (CED)                                        | NeurIPS 2025        | 🔵   | CM       | [[Paper](https://openreview.net/forum?id=NHw8muIAcL)]                                                                                                                                                         |
| 2025.02 | End-to-End Single-Step Flow Matching via Direct Models (FlowFit)                               | ICLR 2025           | 🟢   | FS       | [[Paper](https://openreview.net/forum?id=XpOnko2c59)]                                                                                                                                                         |
| 2024.12 | Adversarial Score identity Distillation (SiDA)                                                 | ICLR 2025           | 🔵   | DDM      | [[Paper](https://arxiv.org/abs/2410.14919)] [[Code](https://github.com/mingyuanzhou/SiD/tree/sida)]                                                                                                           |
| 2024.12 | EM Distillation for One-step Diffusion Models                                                  | NeurIPS 2024        | 🔵   | DDM      | [[Paper](https://arxiv.org/abs/2405.16852)]                                                                                                                                                                   |
| 2024.12 | Flash Diffusion: Accelerating Any Conditional Diffusion Model for Few Steps                    | AAAI 2024           | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2406.02347)] [[Code](https://github.com/gojasper/flash-diffusion)]                                                                                                             |
| 2024.11 | Optimal Flow Matching: Learning Straight Trajectories in Just One Step                         | NeurIPS 2024        | 🟢   | FS       | [[Paper](https://arxiv.org/abs/2403.13117)] [[Code](https://github.com/Jhomanik/Optimal-Flow-Matching)]                                                                                                       |
| 2024.11 | SwiftBrush: One-Step T2I Diffusion with Variational Score Distillation                         | CVPR 2024           | 🔵   | DDM      | [[Paper](https://arxiv.org/abs/2312.05239)] [[Code](https://github.com/VinAIResearch/SwiftBrush)]                                                                                                             |
| 2024.11 | OSDFace: One-Step Diffusion Model for Face Restoration                                         | CVPR 2025           | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2411.17163)] [[Code](https://github.com/jkwang28/OSDFace)]                                                                                                                     |
| 2024.10 | Consistency Models Made Easy                                                                   | ICLR 2025           | 🟢   | CM       | [[Paper](https://arxiv.org/abs/2406.14548)]                                                                                                                                                                   |
| 2024.10 | One-Step Effective Diffusion Network for Real-World Image SR (OSEDiff)                         | NeurIPS 2024        | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2406.08177)] [[Code](https://github.com/cswry/OSEDiff)]                                                                                                                        |
| 2024.10 | SF-V: Single Forward Video Generation Model                                                    | NeurIPS 2024        | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2406.04324)] [[Code](https://github.com/snap-research/SF-V)]                                                                                                                   |
| 2024.10 | Improving the Training of Rectified Flows                                                      | NeurIPS 2024        | 🟢   | FS       | [[Paper](https://arxiv.org/abs/2405.20320)] [[Code](https://github.com/sangyun884/rfpp)]                                                                                                                      |
| 2024.10 | Reward Guided Latent Consistency Distillation (RG-LCD)                                         | ICLR 2025           | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2403.11027)]                                                                                                                                                                   |
| 2024.10 | One-step Diffusion with Distribution Matching Distillation (DMD)                               | CVPR 2024           | 🔵   | DDM      | [[Paper](https://arxiv.org/abs/2311.18828)]                                                                                                                                                                   |
| 2024.10 | One-Step Diffusion Distillation through Score Implicit Matching (SIM)                          | NeurIPS 2024        | 🔵   | DDM      | [[Paper](https://arxiv.org/abs/2410.16794)] [[Code](https://github.com/maple-research-lab/SIM)]                                                                                                               |
| 2024.10 | OneDP: Fast Visuomotor Policies via Diffusion Distillation                                     | ICML 2025           | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2410.21257)]                                                                                                                                                                   |
| 2024.10 | Di[M]O: Distilling Masked Diffusion into One-step                                              | ICCV 2025           | 🔵   | SC       | [[Paper](https://arxiv.org/abs/2503.15457)] [[Code](https://github.com/yuanzhi-zhu/DiMO)]                                                                                                                     |
| 2024.10 | Truncated Consistency Models                                                                   | 2024                | 🟢   | CM       | [[Paper](https://openreview.net/forum?id=ZYDEJEvCbv)]                                                                                                                                                         |
| 2024.09 | Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models                             | ICLR 2025           | 🟢   | APP      | [[Paper](https://arxiv.org/abs/2409.08861)]                                                                                                                                                                   |
| 2024.09 | DKDM: Data-Free Knowledge Distillation for Diffusion Models                                    | CVPR 2025           | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2409.03550)] [[Code](https://github.com/qianlong0502/DKDM)]                                                                                                                    |
| 2024.08 | One Step Diffusion-based SR with Time-Aware Distillation (TAD-SR)                              | NeurIPS 2024        | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2408.07476)] [[Code](https://github.com/LearningHx/TAD-SR)]                                                                                                                    |
| 2024.08 | SwiftBrush v2: Make Your One-step Model Better Than Its Teacher                                | ECCV 2024           | 🔵   | DDM      | [[Paper](https://arxiv.org/abs/2408.14176)] [[Code](https://github.com/vinairesearch/swiftbrushv2)]                                                                                                           |
| 2024.07 | Training Smaller One-Step Diffusion Models with Rectified Flow (SlimFlow)                      | ECCV 2024           | 🔵   | FS       | [[Paper](https://arxiv.org/abs/2407.12718)] [[Code](https://github.com/yuanzhi-zhu/SlimFlow)]                                                                                                                 |
| 2024.06 | Motion Consistency Model (MCM)                                                                 | NeurIPS 2024        | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2406.06890)] [[Code](https://github.com/yhZhai/mcm)]                                                                                                                           |
| 2024.05 | Improved Distribution Matching Distillation (DMD2)                                             | NeurIPS 2024 Oral   | 🔵   | DDM      | [[Paper](https://arxiv.org/abs/2405.14867)] [[Code](https://github.com/tianweiy/dmd2)]                                                                                                                        |
| 2024.05 | Diffusion2GAN: Distilling Diffusion Models into Conditional GANs                               | ECCV 2024           | 🔵   | APP      | [[Paper](https://arxiv.org/abs/2405.05967)]                                                                                                                                                                   |
| 2024.05 | PeRFlow: Piecewise Rectified Flow                                                              | NeurIPS 2024        | 🔵   | FS       | [[Paper](https://arxiv.org/abs/2405.07510)] [[Code](https://github.com/magic-research/piecewise-rectified-flow)]                                                                                              |
| 2024.05 | Phased Consistency Models (PCM)                                                                | NeurIPS 2024        | 🔵   | CM       | [[Paper](https://arxiv.org/abs/2405.18407)] [[Code](https://github.com/G-U-N/Phased-Consistency-Model)]                                                                                                       |
| 2024.04 | Score identity Distillation (SiD)                                                              | ICML 2024           | 🔵   | DDM      | [[Paper](https://arxiv.org/abs/2404.04057)] [[Code](https://github.com/mingyuanzhou/SiD)]                                                                                                                     |
| 2024.04 | Hyper-SD: Trajectory Segmented Consistency Model                                               | NeurIPS 2024        | 🔵   | CM       | [[Paper](https://arxiv.org/abs/2404.13686)] [[Code](https://huggingface.co/ByteDance/Hyper-SD)]                                                                                                               |
| 2024.03 | Consistency Trajectory Models (CTM)                                                            | ICLR 2024           | 🔵   | CM       | [[Paper](https://arxiv.org/abs/2310.02279)] [[Code](https://github.com/sony/ctm)]                                                                                                                             |
| 2024.03 | InstaFlow: One Step is Enough for High-Quality T2I Generation                                  | ICLR 2024           | 🔵   | FS       | [[Paper](https://arxiv.org/abs/2309.06380)] [[Code](https://github.com/gnobitab/InstaFlow)]                                                                                                                   |
| 2024.03 | Latent Adversarial Diffusion Distillation (LADD)                                               | ECCV 2024           | 🔵   | ADV      | [[Paper](https://arxiv.org/abs/2403.12015)]                                                                                                                                                                   |
| 2024.02 | SDXL-Lightning: Progressive Adversarial Diffusion Distillation                                 | 2024                | 🔵   | ADV      | [[Paper](https://arxiv.org/abs/2402.13929)] [[Code](https://huggingface.co/ByteDance/SDXL-Lightning)]                                                                                                         |
| 2024.01 | Rectified Diffusion: Straightness Is Not Your Need                                             | ICLR 2025           | 🟢   | FS       | [[Code](https://github.com/G-U-N/Rectified-Diffusion)]                                                                                                                                                        |
| 2023.11 | Adversarial Diffusion Distillation (ADD)                                                       | ECCV 2024           | 🔵   | ADV      | [[Paper](https://arxiv.org/abs/2311.17042)]                                                                                                                                                                   |
| 2023.10 | Improved Techniques for Training Consistency Models (iCT)                                      | ICLR 2024           | 🟢   | CM       | [[Paper](https://arxiv.org/abs/2310.14189)]                                                                                                                                                                   |
| 2023.10 | Latent Consistency Models (LCM)                                                                | ICLR 2024           | 🔵   | CM       | [[Paper](https://arxiv.org/abs/2310.04378)]                                                                                                                                                                   |
| 2023.06 | BOOT: Data-free Distillation of Denoising Diffusion Models with Bootstrapping                  | ICML 2023           | 🔵   | DDM      | [[Paper](https://arxiv.org/abs/2306.05544)]                                                                                                                                                                   |
| 2023.05 | Diff-Instruct: Universal Knowledge Transfer from Pre-trained Diffusion Models                  | NeurIPS 2023        | 🔵   | DDM      | [[Paper](https://arxiv.org/abs/2305.18455)]                                                                                                                                                                   |
| 2023.03 | Consistency Models                                                                             | ICML 2023           | 🟡   | CM       | [[Paper](https://arxiv.org/abs/2303.01469)]                                                                                                                                                                   |
| 2023.03 | TRACT: Transitive Closure Time-Distillation                                                    | 2023                | 🔵   | FS       | [[Paper](https://arxiv.org/abs/2303.04248)] [[Code](https://github.com/apple/ml-tract)]                                                                                                                       |
| 2022.10 | Flow Matching for Generative Modeling                                                          | ICLR 2023           | 🟢   | FW       | [[Paper](https://arxiv.org/abs/2210.02747)]                                                                                                                                                                   |
| 2022.09 | Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow             | ICLR 2023 Spotlight | 🟢   | FW       | [[Paper](https://arxiv.org/abs/2209.03003)]                                                                                                                                                                   |
| 2022.02 | Progressive Distillation for Fast Sampling of Diffusion Models                                 | ICLR 2023           | 🔵   | FW       | [[Paper](https://arxiv.org/abs/2202.00512)]                                                                                                                                                                   |

---

## Related Resources

### Foundational Works

> These papers are **not** one-step generation methods themselves, but provide the theoretical foundations upon which many one-step methods are built.

- **Progressive Distillation for Fast Sampling of Diffusion Models** [ICLR 2023] 🔵  
[[Paper](https://arxiv.org/abs/2202.00512)]  
Iterative distillation that progressively halves the number of sampling steps; foundational for step-compression research.
- **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow** [ICLR 2023 Spotlight] 🟢  
[[Paper](https://arxiv.org/abs/2209.03003)]  
Learning ODEs that follow straight paths; the reflow procedure inspired many one-step flow methods.
- **Flow Matching for Generative Modeling** [ICLR 2023] 🟢  
[[Paper](https://arxiv.org/abs/2210.02747)]  
Simulation-free training of continuous normalizing flows; the foundational framework behind MeanFlow, SoFlow, etc.

---

## Contributing

Contributions are very welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for paper-entry formatting, category rules, and the native-vs-distillation labeling convention before submitting a pull request.

When adding a paper, classify it by its **primary mathematical contribution** (the principle that makes one-step generation work), then mark it 🟢 native, 🔵 distillation, or 🟡 hybrid, and place it chronologically within the category.

---

## Citation

If you find this list useful in your research, please consider citing it:

```bibtex
@misc{awesome-one-step-generation,
  title        = {Awesome One-Step Generation},
  howpublished = {\url{https://github.com/Luciferbobo/Awesome-One-Step-Generation}},
  year         = {2026}
}
```
