---
layout: default
lang: zh-CN
title: "基于 Cola DLM 的统一多模态 Flow Matching"
description: "基于 Cola DLM 的统一文本-视觉建模"
thumbnail: /assets/fig-unified-overview.png
permalink: /blog/2026/unified-cola/
view_count_offset: 1000
---
<nav class="post-actions" aria-label="文章链接">
  <a href="{{ '/blog/2026/unified-cola-en/' | relative_url }}">English</a>
  <span aria-hidden="true">|</span>
  <span>中文</span>
  <span aria-hidden="true">|</span>
  <a href="https://arxiv.org/abs/2605.06548">Paper</a>
  <span aria-hidden="true">|</span>
  <a href="https://hongcanguo.github.io/Cola-DLM/">Project</a>
</nav>

# 基于 Cola DLM 的统一多模态 Flow Matching

<div class="post-meta">
  <span>发布日期：2026-06-09</span>
  <span>字数：约 6,900</span>
  <span>阅读时间：约 14 分钟</span>
  <span id="busuanzi_container_page_pv">浏览量：<span id="busuanzi_value_page_pv" data-view-count-offset="{{ page.view_count_offset | default: 0 }}">加载中</span></span>
</div>

<details class="toc-card" open markdown="1">
<summary>目录</summary>

- [摘要](#abstract)
- [1. 动机：为什么需要新的统一建模路线？](#motivation)
  - [Cola DLM 回顾](#cola-dlm-recap)
- [2. 连续潜变量上的联合分布](#joint-latents)
- [3. 架构与训练](#architecture-training)
  - [序列排列](#sequence-layout)
  - [注意力掩码作为任务语义的代理](#attention-masks)
  - [目标函数](#objective)
  - [训练设置](#training-setup)
- [4. 定性结果](#qualitative-results)
  - [4.1 文本 → 图像](#text-to-image)
  - [4.2 图像 × 文本 → 文本](#image-text-to-text)
  - [4.3 文本 → 文本](#text-to-text)
- [5. 后续实验：从可行性到定量对比](#future-experiments)
- [6. 路线图：共享联合先验下的更多模态](#roadmap)
- [References](#references)
</details>

<div class="post-note" markdown="1">
**阅读提示。** 本文从动机与统一潜变量建模开始，随后介绍训练结构和定性结果；如果只想快速浏览，可优先阅读第 1、3、4 节。
</div>

### 摘要 {#abstract}
> 最近，统一多模态预训练模型已从早期的单一自回归序列建模，发展为由 Reasoner Tower 与 Generator Tower 组成的双塔范式，包括 MLLM-to-Diffusion 的串联桥接结构，以及基于 Mixture-of-Transformer（MoT）的双塔并联结构。本文探索一种新的基于 **Cola DLM**（连续潜变量扩散语言模型）[1] 的统一建模方案：将文本与视觉信号映射到连续潜变量空间，并使用共享块因果 MMDiT 参数化潜变量生成分布，在统一接口下同时学习理解（text output）与生成（pixel output）。本方法将理解与生成视为同一多模态联合分布的不同条件视角，并通过多任务联合预训练在共享生成分布中同时学习语义表示、跨模态对齐与生成动力学，增强不同模态之间的互信息约束，从而为理解与生成任务提供一种协同机制。本文系统阐述该架构及其关键设计，并展示 **文本→文本**、**文本→图像**、**图像×文本→文本** 三类预训练任务的定性结果。
<!-- 当前实验定位为概念验证（Proof of Concept）：我们验证该统一架构的可行性与初步收敛行为，尚未进行大规模 SFT 或 RL 后训练。 -->

---

## 1. 动机：为什么需要新的统一建模路线？ {#motivation}

近年来，统一多模态预训练建模的研究重点已从「能否用单一系统同时支持理解与生成」，转向「理解路径与生成路径应如何交互和协同」。从架构角度看，现有方法可概括为三类路线。

**路线一：统一自回归序列建模。** Chameleon [2] 将文本与图像均离散化为 token，并在同一 Transformer 中进行预测；Janus / Janus-Pro [3,4] 进一步解耦视觉理解编码器（如 SigLIP [5]）与离散图像 tokenizer（VQ-VAE [6]），但其生成路径仍主要依赖离散视觉 token 的自回归建模。该路线与语言模型范式最为一致，能够以统一序列形式支持交错形式的输入输出；其局限在于图像生成质量和采样效率通常受制于离散 tokenizer 与长序列自回归生成。

**路线二：Reasoner–Generator 双塔并联交互。** Transfusion [7] 在同一多模态模型中联合训练文本 next-token prediction 与图像 diffusion 目标；BAGEL [8] 采用 MoT 架构，使理解专家与生成专家在同一多模态序列上工作，并通过共享 self-attention 实现长上下文交互。Show-o / Show-o2 [9,10] 进一步探索在单一 Transformer 中结合自回归文本建模与离散 diffusion 视觉生成；Tuna / Tuna-2 [11,12] 则从统一视觉表示出发，分别探索连续视觉表示与 pixel embedding 的端到端统一建模。此类方法保留文本侧的 AR 建模，同时在视觉侧引入 diffusion head，相比纯 AR 架构更适合高保真视觉生成；相应地，它需要在表征学习、任务配比以及生成/理解目标之间处理更复杂的优化耦合。

**路线三：Reasoner-to-Generator 串联桥接。** MetaQueries [13] 使用可学习 query 从预训练 MLLM 中提取生成条件，并经由 connector 输入 MMDiT；Qwen-Image [14] 使用 Qwen2.5-VL 提供语义条件，并结合 VAE 与 MMDiT 完成图像生成与编辑；UniWorld-V1 [15]、OmniGen2 [16] 和 UniVideo [17] 同样采用类似的思路，将 MLLM 的理解能力转化为 MMDiT 的条件信号，并扩展至图像编辑、in-context 生成以及视频生成与编辑。该路线具有较强工程可扩展性，能够充分复用已有 MLLM 与 MMDiT 基座。理解与生成通常通过 hidden states、query、connector 或双流条件进行连接，其统一性更多体现为系统级组合，而非由单一模型直接建模多模态联合分布。

上述研究表明，「统一」既不能简单等同于共享一个 Transformer，也不能仅理解为在语言模型之后接入扩散生成模块。即使视觉生成已经发生在 VAE latent 上，大多数现有方法中的跨模态学习仍更接近条件建模，即以一种模态作为另一种模态的条件，而非显式刻画多模态潜变量的联合分布。
基于上述观察，本文旨在构建一种区别于双塔并联或串联桥接的统一训练范式：文本与视觉信号经由 VAE 映射到连续 latent space，并由单一共享的块因果 MMDiT [18,19] 建模

$$
p_\psi\big(z_0^{\text{text}}, z_0^{\text{pixel}}\big).
$$

在该视角下，理解（text output）和生成（pixel output）对应同一多模态联合分布的不同条件视角。多任务联合训练从多个条件方向约束 $p_\psi(z_0^{\text{text}}, z_0^{\text{pixel}})$，促使模型捕获 $Z^{\text{text}}$ 与 $Z^{\text{pixel}}$ 之间的互信息，并在统一接口下共享语义表示、跨模态对齐与生成动力学。
Cola DLM 为文本生成提供了完善的连续潜变量建模方案。本文进一步说明，该方案可自然扩展至其他模态，并为上述统一训练范式提供基础。

### Cola DLM 回顾 {#cola-dlm-recap}

Cola DLM 是一种连续潜变量扩散语言模型。其核心是不在 token 层面执行去噪，而是：

1. 用 **Text VAE** 学习稳定的 **文本 ↔ 连续潜变量** 映射；
2. 用 **块因果 DiT** 在连续潜空间建模**全局语义先验**；
3. 通过**条件解码器**流式生成文本。

扩散过程用于**潜变量先验传输**：

$$
z_1 \sim p_1,\qquad z_0 = \Phi^{\psi}_{0\leftarrow 1}(z_1),\qquad x \sim p_\theta(x \mid z_0).
$$

该分解显式区分了**全局语义组织**（连续潜空间）与**局部文本实现**（解码器）。在扩展至其他模态时，这种分工同样关键。

---

## 2. 连续潜变量上的联合分布 {#joint-latents}

统一建模不应只依赖共享骨干网络，也不应把不同模态强行压到同一表征空间。更自然的做法是先把文本和视觉信号分别映射到连续潜变量，再让共享块因果 MMDiT 建模这些潜变量的联合分布。这样，文本语义、视觉内容以及二者之间的对应关系都可以通过同一个接口来处理。

具体而言，本文遵循与 Cola DLM 相同的概率分解。设 $x_{\text{text}}$、$x_{\text{pixel}}$ 为文本与视觉观测，由相应编码器产生连续潜变量：

$$
z_0^{\text{text}} \in \mathcal{Z}_{\text{text}},\quad
z_0^{\text{text}} \sim q_{\phi_{\text{text}}}(z \mid x_{\text{text}}),
\qquad
z_0^{\text{pixel}} \in \mathcal{Z}_{\text{pixel}},\quad
z_0^{\text{pixel}} \sim q_{\phi_{\text{pixel}}}(z \mid x_{\text{pixel}}).
$$

进一步，将文本与视觉 latent 构成**联合潜变量**并建模统一生成过程：

$$
\bar{z}_0 = \big(z_0^{\text{text}},\, z_0^{\text{pixel}}\big),
\qquad
p(x_{\text{text}}, x_{\text{pixel}}, \bar{z}_0) = p_\theta\big(x_{\text{text}}, x_{\text{pixel}} \mid \bar{z}_0\big)\, p_\psi(\bar{z}_0).
$$

**模态对应的 VAE 编解码器**负责在观测空间和潜变量空间之间转换，**共享块因果 MMDiT** 则在 $\mathcal{Z}_{\text{text}}\times\mathcal{Z}_{\text{pixel}}$ 上参数化 $p_\psi(\bar z_0)$，学习文本 latent 与视觉 latent 的联合分布。「先验传输」可作用于多模态潜变量 $\bar{z}_0$：共享块因果 MMDiT 先生成联合潜变量，随后由对应解码器生成文本或 pixel 输出。连续潜变量的作用在于提供统一的建模接口，而不是要求所有模态共享同一种表示。

从联合分布的角度看，训练目标可以写成：

$$
\mathbb{E}[\mathcal{L}_{\text{ELBO}}]
= \mathbb{E}_q\big[\log p_\theta(x_{\text{text}}, x_{\text{pixel}} \mid \bar{z}_0)\big]
- I\big((X_{\text{text}}, X_{\text{pixel}});\, \bar{Z}_0\big)
- \mathrm{KL}\big(\bar{q}(\bar{z}_0)\,\|\,p_\psi(\bar{z}_0)\big).
$$

这三个项分别对应三个简单的问题：解码器能否从联合潜变量 $\bar z_0$ 还原文本和视觉观测；潜变量中保留了多少关于输入样本的信息；编码器得到的聚合后验 $\bar q(\bar z_0)$ 是否能被共享先验 $p_\psi(\bar z_0)$ 拟合。
在这个框架下，文本 latent 和视觉 latent 不需要被强行对齐成逐点可比的同一种表示。模型能够在同一个联合分布中学习不同的条件生成方向，例如 $p_\psi(z^{\text{pixel}}\mid z^{\text{text}})$，或 $p_\psi(z^{\text{text}}\mid z^{\text{pixel}})$。

因此，潜变量承载压缩后的高层语义，解码器负责把这些语义落实为具体的文本或 pixel 输出。统一建模发生在连续潜变量的联合分布上，而不只是复用同一个跨模态骨干网络。

---

## 3. 架构与训练 {#architecture-training}

![**图 1.** 基于 Cola DLM 的统一文本–视觉建模。左：文本续写与图像描述生成。中：文生图。右：方法示意。文本与视觉信号被映射到连续潜变量，并由共享块因果 MMDiT 建模联合生成分布。]({{ '/assets/fig-unified-overview.png' | relative_url }})

整体架构如图 1 右侧所示，以文本和图像两种模态为例，包含以下组成部分：

- **文本路径。** **Text VAE** 将文本映射为连续潜变量；文本序列被**划分为块**。
- **图像路径。** **Image VAE** 将图像映射为紧凑潜变量；每个图像被视为**单一块**。
- **共享块因果 MMDiT。** 该模块在文本块与图像块上运算，用于参数化联合潜变量生成分布，并支持**模态内**处理与**跨模态**交互。当前图像实例中堆叠约 **1/3 双流**、**2/3 单流** DiT 块。

单一模型在同一框架内支持三类任务：

- **文本 → 文本（T2T）**，
- **文本 → 图像（T2I）**，
- **图像 × 文本 → 文本（IT2T）**（图像描述 / 视觉问答）。

### 序列排列 {#sequence-layout}

具体来说，我们将连续潜变量打包为序列，并为每个位置标注两个整数。设打包序列有 $N$ 个位置，位置 $i$ 携带：

- **段类型** $s_i \in \{\textsf{P}, \textsf{C}, \textsf{I}, \textsf{N}\}$，分别表示*指令文本*、*无噪文本*、*图像*、*带噪文本*；
- **块索引** $b_i \in \mathbb{Z}$，将文本划分为大小为 $B$ 的连续块（当前实现中 $B=16$）。

文本潜变量按块大小 $B$ 切分；图像潜变量视为*单个*块。两种标注承载全部任务语义：

- **无噪文本**（$\textsf{C}$）潜变量为**上下文/条件**，**不计损失**；
- **带噪**（$\textsf{N}$）潜变量为**生成目标**，接受扩散损失；
- **指令**（$\textsf{P}$，仅用于 IT2T 任务）为任务**提示**，同样作为上下文；其与 $\textsf{C}$ 分离，以避免条件与目标混淆（原因见下文 IT2T 部分）；
- **块索引**在文本块间施加**因果顺序**，同时使每个块内部以及单个图像内部保持**双向**注意力。

任务仅由 (i) 各段是否存在及顺序，(ii) 耦合它们的注意力规则定义，三种任务的布局如下：

| 任务 | 每样本段布局 | 扩散目标 |
| --- | --- | --- |
| 文本 → 文本 (T2T) | $[\,\textsf{C}\,]\,[\,\textsf{N}\,]$ | 文本块 |
| 文本 → 图像 (T2I) | $[\,\textsf{C}\,]\,[\,\textsf{I}\,]\,[\,\textsf{N}\,]$ | 图像（主）+ 描述（辅） |
| 图像 × 文本 → 文本 (IT2T) | $[\,\textsf{P}\,]\,[\,\textsf{I}\,]\,[\,\textsf{C}\,]\,[\,\textsf{N}\,]$ | 仅文本（$\textsf{P},\textsf{I}$ 为固定上下文） |

单次前向中，多个样本（梯度累积下可为不同任务）拼接为长序列；注意力额外约束在**同一样本内部**，即位置 $i$ 的 query 仅可 attend 到同一样本的位置 $j$。

### 注意力掩码作为任务语义的代理 {#attention-masks}

共享块因果 MMDiT 的任务语义由注意力掩码 $M_{ij}\in\{0,1\}$ 控制，其决定 query $i$ 是否 attend 到 key $j$。掩码由以下段条件规则给出。**图 2** 为各任务小规模示例渲染得到的掩码（行为 query，列为被 attend 的 key）。

![**图 2.** 共享块因果 MMDiT 在三项任务下的注意力掩码，由下列段条件规则渲染。行为 query，列为被 attend 的 key；红色为可 attend，灰色为屏蔽。顶边与左边的色条编码各位置的段类型（绿 = IT2T 指令段 $\textsf{P}$，蓝 = 无噪文本，橙 = 图像，紫 = 带噪文本）。]({{ '/assets/fig-attention-masks.png' | relative_url }})

**文本 → 图像。** 图像生成以干净文本为条件。另外，额外拼接带噪文本做单一序列内部的多任务联训（T2I+T2T），注意力规则如下：

$$
M_{ij}=1 \iff
\begin{cases}
s_j=\textsf{C}\ \wedge\ b_j\le b_i, & s_i=\textsf{C}\quad(\text{块因果描述}),\\[2pt]
s_j\in\{\textsf{C},\textsf{I}\}, & s_i=\textsf{I}\quad(\text{完整描述条件 + 双向自注意力}),\\[2pt]
(s_j=\textsf{C}\ \wedge\ b_j<b_i)\ \vee\ (s_j=\textsf{N}\ \wedge\ b_j=b_i), & s_i=\textsf{N}\quad(\text{块扩散文本}).
\end{cases}
$$

需要指出，图像段与带噪文本段**互不** attend。因此，两项目标并非通过辅助文本损失直接耦合，而是经由 (i) 图像 attend 干净文本，使文本条件从图像损失获得梯度，以及 (ii) 共享块因果 MMDiT 参数。

**图像 × 文本 → 文本**。**指令** $\textsf{P}$ 与 **图像** $\textsf{I}$ 均为干净条件，模型基于该条件生成文本答案。该任务的关键设计在于，IT2T 中的干净文本流同时包含任务提示与答案的 teacher-forcing 副本。若将二者合并为同一段，会导致答案泄漏到图像表示中并被带噪文本读取，从而短路扩散目标；若图像完全不可见提示，则视觉编码又无法感知任务指令。因此，我们将指令置于独立段 $\textsf{P}$，并与描述上下文 $\textsf{C}$ 分离，以解决该问题：

$$
M_{ij}=1 \iff
\begin{cases}
s_j\in\{\textsf{P},\textsf{I}\}, & s_i=\textsf{P}\quad(\text{指令与图像构成双向条件}),\\[2pt]
s_j\in\{\textsf{P},\textsf{I}\}, & s_i=\textsf{I}\quad(\text{图像见指令，不见答案}),\\[2pt]
(s_j=\textsf{C}\ \wedge\ b_j\le b_i)\ \vee\ s_j\in\{\textsf{P},\textsf{I}\}, & s_i=\textsf{C}\quad(\text{答案上下文，以提示与图像为条件}),\\[2pt]
(s_j=\textsf{C}\ \wedge\ b_j<b_i)\ \vee\ (s_j=\textsf{N}\ \wedge\ b_j=b_i)\ \vee\ s_j\in\{\textsf{P},\textsf{I}\}, & s_i=\textsf{N}\quad(\text{块扩散答案，以提示与图像为条件}).
\end{cases}
$$

由此，指令与图像形成单一、全双向的条件块，使视觉编码具备提示感知能力；同时二者均不 attend 答案（$\textsf{C}$ 或 $\textsf{N}$），从而避免答案反向泄漏进条件。答案部分（$\textsf{C}$ 上下文与 $\textsf{N}$ 目标）采用块因果结构，并以 $\{\textsf{P},\textsf{I}\}$ 为条件。由于图像从不作为答案 query 目标，在该任务中图像仅作为上下文，图像扩散损失被关闭。

**文本 → 文本** 可视为去除图像段后的 T2I 规则特例：无噪前缀通过块因果方式编码，其余文本块在先前无噪块条件下由块扩散生成。

### 目标函数 {#objective}

每步损失为各有效目标之和：

$$
\mathcal{L} = \mathcal{L}_{\text{image}} \;+\; \mathcal{L}_{\text{text}} \;+\; \mathcal{L}_{\text{REPA}},
$$

各项均为相应潜空间中的速度预测 MSE。$\mathcal{L}_{\text{REPA}}$ 为可选表示对齐项[20]（仅在 T2I 中激活），用于将 DiT 中间层与视觉编码器特征对齐以加速生成收敛。训练时按可配置任务比例在每个训练步采样任务，并在所有 worker 上同步任务选择，以保证分布式 collective 调用一致。

### 训练设置 {#training-setup}

- **噪声调度。** 两模态均采用 rectified-flow 的线性插值方式，速度预测与 Euler 采样器。
- **块级噪声。** 文本上**每块独立采样一个时间步**，块内所有 token 共享——继承自 Cola DLM 的层次化块因果核心。图像潜变量作为单块加噪。
- **模态专属时间步采样。** 训练时间步按模态从 logit-normal 分布采样：图像块和文本块独立采样。
- **Classifier-free guidance。** 训练时随机丢弃 T2I 样本的整段文本条件，使模型学习无条件视觉生成分布。

所有模块均**从零预训练**，在不同预训练阶段调整任务数据比例。本文报告的是当前 Proof of Concept 规模的预训练设置，未经过大规模指令微调或 RL 后训练的完整系统：

| 任务 | 设置 |
| --- | --- |
| 文本 → 图像 | 256 分辨率 **80k** 步（全局 batch ≈ **3k**），再 640 分辨率 **10k** 步（全局 batch ≈ **1k**） |
| 图像 × 文本 → 文本 | 相同 batch 配置，约 **50k** 步 |
| 文本（合计） | 约 **10 亿** 文本 token |
| 多模态监督 | 约 **500 万** 图文对 |

在上述有限数据与算力设置下，模型已能够生成连贯文本、结构合理的图像，并表现出初步图像描述能力。这一结果表明，先在潜变量层面学习联合分布，再交由各模态解码器生成最终输出，可能是以较高数据效率获得跨模态能力的一种有效途径。

---

## 4. 定性结果 {#qualitative-results}

本节报告我们的模型在三类任务上的定性结果。

### 4.1 文本 → 图像 {#text-to-image}

下面展示若干单图样本，点击任意缩略图可查看大图，并可继续逐张浏览。


<div class="sample-carousel" aria-label="文本到图像样本画廊">
  <div class="sample-track">
    <figure class="sample-slide" id="t2i-sample-1">
      <a class="sample-nav prev" href="#t2i-sample-39" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/56.jpg' | relative_url }}" alt="文本到图像样本 1" />
      <a class="sample-nav next" href="#t2i-sample-2" aria-label="下一张">›</a>
      <figcaption>A delicate white dandelion seed head stands above a green meadow, with other soft, blurred flowers nearby. A dark row of trees and a cloudy blue-gray sky give the open field a calm, breezy spring atmosphere.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-2">
      <a class="sample-nav prev" href="#t2i-sample-1" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/82.jpg' | relative_url }}" alt="文本到图像样本 2" />
      <a class="sample-nav next" href="#t2i-sample-3" aria-label="下一张">›</a>
      <figcaption>Two large abstract paintings dominate a polished interior hallway, their vivid colors contrasting with the neutral wall panels and glossy floor. Ceiling spotlights, patterned rugs, and nearby doorways create the feeling of a private gallery or elegant office space.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-3">
      <a class="sample-nav prev" href="#t2i-sample-2" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/103.jpg' | relative_url }}" alt="文本到图像样本 3" />
      <a class="sample-nav next" href="#t2i-sample-4" aria-label="下一张">›</a>
      <figcaption>Several boats glide across rippling water during sunset, appearing as dark silhouettes beneath a long arched bridge. The orange sun and its reflection stretch across the river, giving the scene a warm, tranquil waterfront mood.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-4">
      <a class="sample-nav prev" href="#t2i-sample-3" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/123.jpg' | relative_url }}" alt="文本到图像样本 4" />
      <a class="sample-nav next" href="#t2i-sample-5" aria-label="下一张">›</a>
      <figcaption>White spring blossoms cluster thickly along interwoven tree branches, mixed with small green leaves and unopened buds. The shallow depth of field makes the foreground flowers stand out while the background turns into a delicate canopy of pale bloom.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-5">
      <a class="sample-nav prev" href="#t2i-sample-4" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/134.jpg' | relative_url }}" alt="文本到图像样本 5" />
      <a class="sample-nav next" href="#t2i-sample-6" aria-label="下一张">›</a>
      <figcaption>Golden sunset light spreads behind a line of dark mountain silhouettes and reflects across the still lake below. Grasses in the foreground add depth, while the mirrored peaks and warm sky create a peaceful landscape composition.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-6">
      <a class="sample-nav prev" href="#t2i-sample-5" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/280.jpg' | relative_url }}" alt="文本到图像样本 6" />
      <a class="sample-nav next" href="#t2i-sample-7" aria-label="下一张">›</a>
      <figcaption>A broad blue lake stretches toward layered mountains beneath a sweeping sky of textured white clouds. The calm ripples, low dark shorelines, and distant ridges create a cool, expansive landscape with a quiet sense of depth.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-7">
      <a class="sample-nav prev" href="#t2i-sample-6" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/h1.jpg' | relative_url }}" alt="文本到图像样本 7" />
      <a class="sample-nav next" href="#t2i-sample-8" aria-label="下一张">›</a>
      <figcaption>A young woman in a white coat and patterned scarf stands at night in front of glowing red and yellow lanterns. The warm lights and wooden structures behind her create a festive, atmospheric portrait with soft contrast.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-8">
      <a class="sample-nav prev" href="#t2i-sample-7" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/l1.jpg' | relative_url }}" alt="文本到图像样本 8" />
      <a class="sample-nav next" href="#t2i-sample-9" aria-label="下一张">›</a>
      <figcaption>A monochrome mountain landscape drawing shows steep ridges, deep valleys, winding water, and layered terrain fading into the distance. The sketch-like shading gives the scene a rugged, hand-rendered quality, as if from a travel notebook.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-9">
      <a class="sample-nav prev" href="#t2i-sample-8" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/67.jpg' | relative_url }}" alt="文本到图像样本 9" />
      <a class="sample-nav next" href="#t2i-sample-10" aria-label="下一张">›</a>
      <figcaption>Tall trees on both sides frame a dramatic sunset sky filled with glowing orange, pale yellow, and blue cloud textures. The dark silhouettes below make the scene feel quiet, spacious, and gently illuminated by evening light.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-10">
      <a class="sample-nav prev" href="#t2i-sample-9" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/83.jpg' | relative_url }}" alt="文本到图像样本 10" />
      <a class="sample-nav next" href="#t2i-sample-11" aria-label="下一张">›</a>
      <figcaption>A cozy outdoor dining area is lit by a large yellow umbrella and several red hanging lanterns. Wooden tables, simple chairs, and warm pools of light create an intimate evening setting with a festive, traditional atmosphere.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-11">
      <a class="sample-nav prev" href="#t2i-sample-10" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/86.jpg' | relative_url }}" alt="文本到图像样本 11" />
      <a class="sample-nav next" href="#t2i-sample-12" aria-label="下一张">›</a>
      <figcaption>A dense column of black and gray smoke rises from a rural property near small buildings and scattered trees. The surrounding dirt paths, fields, and pale sky make the sudden fire or burning event feel stark and unsettling.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-12">
      <a class="sample-nav prev" href="#t2i-sample-11" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/99.jpg' | relative_url }}" alt="文本到图像样本 12" />
      <a class="sample-nav next" href="#t2i-sample-13" aria-label="下一张">›</a>
      <figcaption>A vivid red rose fills the image in extreme close-up, revealing many layered petals curling toward a tight spiral at the center. The blurred green background keeps attention on the flower's saturated color and soft texture.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-13">
      <a class="sample-nav prev" href="#t2i-sample-12" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/275.jpg' | relative_url }}" alt="文本到图像样本 13" />
      <a class="sample-nav next" href="#t2i-sample-14" aria-label="下一张">›</a>
      <figcaption>A bearded man with curly hair sits indoors wearing a dark blazer and lavender shirt. Warm lamp light, stacks of documents, and framed artwork behind him suggest an office or study, giving the portrait a serious conversational tone.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-14">
      <a class="sample-nav prev" href="#t2i-sample-13" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/308.jpg' | relative_url }}" alt="文本到图像样本 14" />
      <a class="sample-nav next" href="#t2i-sample-15" aria-label="下一张">›</a>
      <figcaption>A sightseeing boat moves along a wide river in front of grand historic architecture, including ornate towers, stone facades, and decorative rooflines. Reflections shimmer on the water, emphasizing the scale and elegance of the riverside cityscape.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-15">
      <a class="sample-nav prev" href="#t2i-sample-14" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/323.png' | relative_url }}" alt="文本到图像样本 15" />
      <a class="sample-nav next" href="#t2i-sample-16" aria-label="下一张">›</a>
      <figcaption>A single leafless tree stands prominently in rolling green grassland, its intricate branches spreading against a bright sky. Distant mountains and soft clouds frame the isolated tree, giving the image a quiet, almost sculptural quality.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-16">
      <a class="sample-nav prev" href="#t2i-sample-15" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/344.jpg' | relative_url }}" alt="文本到图像样本 16" />
      <a class="sample-nav next" href="#t2i-sample-17" aria-label="下一张">›</a>
      <figcaption>A simple stylized illustration presents a red rose with green leaves, centered inside a rounded purple background with a thick black outline. The bold colors, clean shapes, and cartoon-like linework make the flower feel decorative and playful.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-17">
      <a class="sample-nav prev" href="#t2i-sample-16" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/411.jpg' | relative_url }}" alt="文本到图像样本 17" />
      <a class="sample-nav next" href="#t2i-sample-18" aria-label="下一张">›</a>
      <figcaption>Bright pink blossoms fill the foreground in dense clusters, their yellow centers catching the light. More branches and flowers blur behind them, creating a saturated spring scene that feels lively, sunny, and full of floral texture.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-18">
      <a class="sample-nav prev" href="#t2i-sample-17" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/417.jpg' | relative_url }}" alt="文本到图像样本 18" />
      <a class="sample-nav next" href="#t2i-sample-19" aria-label="下一张">›</a>
      <figcaption>A loose navy blue sweater hangs from a wooden hanger against a plain white background. The soft fabric, wide sleeves, and relaxed drape are emphasized by the simple product-style composition and uncluttered studio-like setting.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-19">
      <a class="sample-nav prev" href="#t2i-sample-18" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/429.jpg' | relative_url }}" alt="文本到图像样本 19" />
      <a class="sample-nav next" href="#t2i-sample-20" aria-label="下一张">›</a>
      <figcaption>A misty mountain rises beyond a quiet lake, its slopes softened by gray haze and overcast light. Reeds in the foreground frame the water, adding depth to the subdued scene and strengthening its calm, contemplative mood.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-20">
      <a class="sample-nav prev" href="#t2i-sample-19" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/580.jpg' | relative_url }}" alt="文本到图像样本 20" />
      <a class="sample-nav next" href="#t2i-sample-21" aria-label="下一张">›</a>
      <figcaption>A calm lake reflects a bright blue sky and the low green hills beyond the opposite shore. Reeds and leafy plants in the foreground create a natural frame, while distant buildings add a small human presence to the peaceful view.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-21">
      <a class="sample-nav prev" href="#t2i-sample-20" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/591.jpg' | relative_url }}" alt="文本到图像样本 21" />
      <a class="sample-nav next" href="#t2i-sample-22" aria-label="下一张">›</a>
      <figcaption>A roadside viewpoint looks out across blue ocean water toward distant islands and hazy mountains. Green shrubs, tall grasses, a utility pole, and a small signboard sit beside the path, giving the scenic overlook a casual travel feeling.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-22">
      <a class="sample-nav prev" href="#t2i-sample-21" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/641.jpg' | relative_url }}" alt="文本到图像样本 22" />
      <a class="sample-nav next" href="#t2i-sample-23" aria-label="下一张">›</a>
      <figcaption>A cup of foamy coffee sits beside fried pastries sprinkled with powdered sugar on brown paper. The colorful table covering, golden snacks, and creamy drink create a cheerful cafe or street-food scene focused on sweet comfort food.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-23">
      <a class="sample-nav prev" href="#t2i-sample-22" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/689.jpeg' | relative_url }}" alt="文本到图像样本 23" />
      <a class="sample-nav next" href="#t2i-sample-24" aria-label="下一张">›</a>
      <figcaption>Heavy gray storm clouds gather low over city buildings, power lines, and rooftops, filling most of the frame with dark, textured sky. The urban skyline beneath looks small and tense, suggesting an approaching downpour or dramatic weather change.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-24">
      <a class="sample-nav prev" href="#t2i-sample-23" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/696.jpg' | relative_url }}" alt="文本到图像样本 24" />
      <a class="sample-nav next" href="#t2i-sample-25" aria-label="下一张">›</a>
      <figcaption>A large steamed dumpling or bun rests on a round wooden plate, garnished with lettuce and small red tomatoes. The bright tabletop, folded cloth, and clean plating make the simple food presentation look fresh and carefully arranged.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-25">
      <a class="sample-nav prev" href="#t2i-sample-24" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/721.jpg' | relative_url }}" alt="文本到图像样本 25" />
      <a class="sample-nav next" href="#t2i-sample-26" aria-label="下一张">›</a>
      <figcaption>A sculptural bust shows a realistic human face emerging from a rough, crown-like form and textured reddish base. The plain wall behind it keeps focus on the unusual material contrast, facial detail, and handmade artistic expression.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-26">
      <a class="sample-nav prev" href="#t2i-sample-25" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/760.jpg' | relative_url }}" alt="文本到图像样本 26" />
      <a class="sample-nav next" href="#t2i-sample-27" aria-label="下一张">›</a>
      <figcaption>A round celebration cake is decorated with pink frosting roses, green leaves, and a large bow across the top. Plates of food around it suggest a shared meal, while the elaborate icing makes the cake the table's centerpiece.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-27">
      <a class="sample-nav prev" href="#t2i-sample-26" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/765.jpg' | relative_url }}" alt="文本到图像样本 27" />
      <a class="sample-nav next" href="#t2i-sample-28" aria-label="下一张">›</a>
      <figcaption>Rows of grilled skewers are packed tightly on a metal tray, coated in glossy red seasoning, herbs, and chili flakes. The close-up view highlights the rich sauce, charred edges, and spicy street-food appeal of the dish.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-28">
      <a class="sample-nav prev" href="#t2i-sample-27" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/807.jpg' | relative_url }}" alt="文本到图像样本 28" />
      <a class="sample-nav next" href="#t2i-sample-29" aria-label="下一张">›</a>
      <figcaption>Thick steam rises from a large pot filled with stew, meat, vegetables, and rich brown broth. The close view captures active cooking, with bubbling liquid and softened ingredients creating a hot, hearty, and aromatic kitchen scene.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-29">
      <a class="sample-nav prev" href="#t2i-sample-28" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/808.jpg' | relative_url }}" alt="文本到图像样本 29" />
      <a class="sample-nav next" href="#t2i-sample-30" aria-label="下一张">›</a>
      <figcaption>Small white flowers bloom in dense clusters among glossy green leaves, with many speckled petals and unopened buds visible. The close-up composition emphasizes delicate natural patterns, soft lighting, and the fresh detail of the flowering plant.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-30">
      <a class="sample-nav prev" href="#t2i-sample-29" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/850.jpg' | relative_url }}" alt="文本到图像样本 30" />
      <a class="sample-nav next" href="#t2i-sample-31" aria-label="下一张">›</a>
      <figcaption>A brilliant sun shines above a winding mountain road bordered by lush green plants. Distant ridges fade toward the blue horizon, while the intense light and clear sky create a bright, open, high-altitude travel scene.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-31">
      <a class="sample-nav prev" href="#t2i-sample-30" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/884.jpg' | relative_url }}" alt="文本到图像样本 31" />
      <a class="sample-nav next" href="#t2i-sample-32" aria-label="下一张">›</a>
      <figcaption>A vivid red boat rests on pale sand in front of calm turquoise water and a clear horizon. The simple composition emphasizes strong color contrast, seaside quietness, and the unusual stillness of a boat pulled ashore.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-32">
      <a class="sample-nav prev" href="#t2i-sample-31" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/h2.jpg' | relative_url }}" alt="文本到图像样本 32" />
      <a class="sample-nav next" href="#t2i-sample-33" aria-label="下一张">›</a>
      <figcaption>A man in a gray polo shirt stands on a balcony or rooftop, looking thoughtfully to the side. A blurred city skyline behind him gives the portrait an urban setting and a calm, reflective mood.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-33">
      <a class="sample-nav prev" href="#t2i-sample-32" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/h4.jpg' | relative_url }}" alt="文本到图像样本 33" />
      <a class="sample-nav next" href="#t2i-sample-34" aria-label="下一张">›</a>
      <figcaption>A close selfie shows a man wearing a dark bucket hat and light blue shirt while seated inside a vehicle. Black seats, side windows, and an orange curtain provide context for the casual travel setting.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-34">
      <a class="sample-nav prev" href="#t2i-sample-33" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/h5.jpg' | relative_url }}" alt="文本到图像样本 34" />
      <a class="sample-nav next" href="#t2i-sample-35" aria-label="下一张">›</a>
      <figcaption>An anime-style girl with long dark hair wears a purple patterned outfit with lace details, posed against a dreamy sky. Large moonlike circles, stars, and soft gradients create a romantic fantasy illustration style.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-35">
      <a class="sample-nav prev" href="#t2i-sample-34" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/h6.jpg' | relative_url }}" alt="文本到图像样本 35" />
      <a class="sample-nav next" href="#t2i-sample-36" aria-label="下一张">›</a>
      <figcaption>A woman with curly hair smiles gently in a very dark outdoor setting, lit softly from the front. The nearly black background isolates her face and shoulders, creating an intimate portrait with understated nighttime atmosphere.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-36">
      <a class="sample-nav prev" href="#t2i-sample-35" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/h7.jpg' | relative_url }}" alt="文本到图像样本 36" />
      <a class="sample-nav next" href="#t2i-sample-37" aria-label="下一张">›</a>
      <figcaption>Two men in gray shirts stand shoulder to shoulder in front of a green screen. One smiles broadly while the other gives a restrained expression, creating a casual studio portrait with contrasting personalities.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-37">
      <a class="sample-nav prev" href="#t2i-sample-36" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/h8.jpg' | relative_url }}" alt="文本到图像样本 37" />
      <a class="sample-nav next" href="#t2i-sample-38" aria-label="下一张">›</a>
      <figcaption>Two adults sit at a restaurant table behind a birthday cake topped with lit candles. Warm indoor lighting, surrounding dishes, cups, and desserts suggest a celebratory meal shared in a relaxed dining setting.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-38">
      <a class="sample-nav prev" href="#t2i-sample-37" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/h9.jpg' | relative_url }}" alt="文本到图像样本 38" />
      <a class="sample-nav next" href="#t2i-sample-39" aria-label="下一张">›</a>
      <figcaption>A fluffy white cat sits indoors with wide dark eyes, pink ears, and crossed front paws. The softly blurred background and centered framing make the cat's round face and clean white fur especially prominent.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-39">
      <a class="sample-nav prev" href="#t2i-sample-38" aria-label="上一张">‹</a>
      <img src="{{ '/assets/t2i_samples/h10.jpg' | relative_url }}" alt="文本到图像样本 39" />
      <a class="sample-nav next" href="#t2i-sample-1" aria-label="下一张">›</a>
      <figcaption>Two young women in coordinated school-style outfits with red bows take a close selfie outdoors. Trees, bicycles, and paved walkways in the background suggest a campus or park setting on a bright day.</figcaption>
    </figure>
  </div>
  <div class="sample-dots" aria-label="文本到图像样本快速跳转">
    <a href="#t2i-sample-1">1</a>
    <a href="#t2i-sample-2">2</a>
    <a href="#t2i-sample-3">3</a>
    <a href="#t2i-sample-4">4</a>
    <a href="#t2i-sample-5">5</a>
    <a href="#t2i-sample-6">6</a>
    <a href="#t2i-sample-7">7</a>
    <a href="#t2i-sample-8">8</a>
    <a href="#t2i-sample-9">9</a>
    <a href="#t2i-sample-10">10</a>
    <a href="#t2i-sample-11">11</a>
    <a href="#t2i-sample-12">12</a>
    <a href="#t2i-sample-13">13</a>
    <a href="#t2i-sample-14">14</a>
    <a href="#t2i-sample-15">15</a>
    <a href="#t2i-sample-16">16</a>
    <a href="#t2i-sample-17">17</a>
    <a href="#t2i-sample-18">18</a>
    <a href="#t2i-sample-19">19</a>
    <a href="#t2i-sample-20">20</a>
    <a href="#t2i-sample-21">21</a>
    <a href="#t2i-sample-22">22</a>
    <a href="#t2i-sample-23">23</a>
    <a href="#t2i-sample-24">24</a>
    <a href="#t2i-sample-25">25</a>
    <a href="#t2i-sample-26">26</a>
    <a href="#t2i-sample-27">27</a>
    <a href="#t2i-sample-28">28</a>
    <a href="#t2i-sample-29">29</a>
    <a href="#t2i-sample-30">30</a>
    <a href="#t2i-sample-31">31</a>
    <a href="#t2i-sample-32">32</a>
    <a href="#t2i-sample-33">33</a>
    <a href="#t2i-sample-34">34</a>
    <a href="#t2i-sample-35">35</a>
    <a href="#t2i-sample-36">36</a>
    <a href="#t2i-sample-37">37</a>
    <a href="#t2i-sample-38">38</a>
    <a href="#t2i-sample-39">39</a>
  </div>
</div>

### 4.2 图像 × 文本 → 文本 {#image-text-to-text}

在训练约 **500 万** 图文对后，模型表现出初步的图像描述能力，能够根据图像内容生成相应文本。

<div class="sample-carousel" aria-label="图像条件文本生成样本画廊">
  <div class="sample-track">
    <figure class="sample-slide" id="it2t-sample-1">
      <a class="sample-nav prev" href="#it2t-sample-36" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/8.png' | relative_url }}" alt="图像条件文本生成样本 1" />
      <a class="sample-nav next" href="#it2t-sample-2" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a woman wearing an oversized long down jacket with a plain collar. The jacket features a letter ``A'' design on it.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-2">
      <a class="sample-nav prev" href="#it2t-sample-1" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/11.png' | relative_url }}" alt="图像条件文本生成样本 2" />
      <a class="sample-nav next" href="#it2t-sample-3" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a person wearing a black jacket and sunglasses, standing on a street.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-3">
      <a class="sample-nav prev" href="#it2t-sample-2" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/54.png' | relative_url }}" alt="图像条件文本生成样本 3" />
      <a class="sample-nav next" href="#it2t-sample-4" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a meeting room with several people, around six to eight, sitting around a long conference table and holding a meeting. A screen is mounted on the wall, and bowls, cups, and pens are placed on the table.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-4">
      <a class="sample-nav prev" href="#it2t-sample-3" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/57.png' | relative_url }}" alt="图像条件文本生成样本 4" />
      <a class="sample-nav next" href="#it2t-sample-5" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a person wearing a grey hoodie with a purple design on the front. The background includes a white wall with a blue design.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-5">
      <a class="sample-nav prev" href="#it2t-sample-4" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/58.png' | relative_url }}" alt="图像条件文本生成样本 5" />
      <a class="sample-nav next" href="#it2t-sample-6" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows several people playing a ball game on a court. The background includes a blue wall with some text, and the ground is also blue.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-6">
      <a class="sample-nav prev" href="#it2t-sample-5" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/59.png' | relative_url }}" alt="图像条件文本生成样本 6" />
      <a class="sample-nav next" href="#it2t-sample-7" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a person wearing a black suit with a white shirt underneath, seated.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-7">
      <a class="sample-nav prev" href="#it2t-sample-6" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/60.png' | relative_url }}" alt="图像条件文本生成样本 7" />
      <a class="sample-nav next" href="#it2t-sample-8" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a cluster of ancient buildings with old-style architecture, featuring several old structures packed closely together.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-8">
      <a class="sample-nav prev" href="#it2t-sample-7" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/61.png' | relative_url }}" alt="图像条件文本生成样本 8" />
      <a class="sample-nav next" href="#it2t-sample-9" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image is a cityscape shot at dusk, showing an urban night scene with many tall high-rise buildings.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-9">
      <a class="sample-nav prev" href="#it2t-sample-8" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/84.png' | relative_url }}" alt="图像条件文本生成样本 9" />
      <a class="sample-nav next" href="#it2t-sample-10" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a large football (soccer) stadium with the field inside, resembling an outdoor stadium. Another row of stands is visible on the other side of the field.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-10">
      <a class="sample-nav prev" href="#it2t-sample-9" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/89.png' | relative_url }}" alt="图像条件文本生成样本 10" />
      <a class="sample-nav next" href="#it2t-sample-11" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a person using a pen to write on a piece of paper.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-11">
      <a class="sample-nav prev" href="#it2t-sample-10" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/12.png' | relative_url }}" alt="图像条件文本生成样本 11" />
      <a class="sample-nav next" href="#it2t-sample-12" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> What does this image depict? Give a comprehensive description.<br /><strong>Output:</strong> The image shows a woman with long, dark hair wearing a black top. The background is a neutral color, with a window and a pink wall visible behind her. She appears to be standing indoors.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-12">
      <a class="sample-nav prev" href="#it2t-sample-11" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/13.png' | relative_url }}" alt="图像条件文本生成样本 12" />
      <a class="sample-nav next" href="#it2t-sample-13" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> What does this image depict? Give a comprehensive description.<br /><strong>Output:</strong> The image shows a portrait of a young woman with blonde hair, wearing a white top and seated at a student desk with an open laptop. She appears to be speaking while using the laptop.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-13">
      <a class="sample-nav prev" href="#it2t-sample-12" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/15.png' | relative_url }}" alt="图像条件文本生成样本 13" />
      <a class="sample-nav next" href="#it2t-sample-14" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> What does this image depict? Give a comprehensive description.<br /><strong>Output:</strong> The image shows a gravestone with English text carved on it.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-14">
      <a class="sample-nav prev" href="#it2t-sample-13" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/36.png' | relative_url }}" alt="图像条件文本生成样本 14" />
      <a class="sample-nav next" href="#it2t-sample-15" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image shows a cluster of modern city buildings, including tall skyscrapers with distinctive and futuristic architectural designs.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-15">
      <a class="sample-nav prev" href="#it2t-sample-14" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/43.png' | relative_url }}" alt="图像条件文本生成样本 15" />
      <a class="sample-nav next" href="#it2t-sample-16" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image is a medium shot of a young East Asian woman with long, straight dark hair, wearing a pink top and a necklace. She is posing with her hands raised near her face, making a gesture with her fingers.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-16">
      <a class="sample-nav prev" href="#it2t-sample-15" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/44.png' | relative_url }}" alt="图像条件文本生成样本 16" />
      <a class="sample-nav next" href="#it2t-sample-17" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image shows a blue card or cover with a printed image and some text on it.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-17">
      <a class="sample-nav prev" href="#it2t-sample-16" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/46.png' | relative_url }}" alt="图像条件文本生成样本 17" />
      <a class="sample-nav next" href="#it2t-sample-18" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image shows three people standing in an open outdoor area. They are wearing colored clothing, and one of them appears to be carrying something on their back.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-18">
      <a class="sample-nav prev" href="#it2t-sample-17" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/67.png' | relative_url }}" alt="图像条件文本生成样本 18" />
      <a class="sample-nav next" href="#it2t-sample-19" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image shows a mountain range with a massive peak. The surface of the mountain reveals distinctive textures and patterns.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-19">
      <a class="sample-nav prev" href="#it2t-sample-18" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/68.png' | relative_url }}" alt="图像条件文本生成样本 19" />
      <a class="sample-nav next" href="#it2t-sample-20" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image shows a man wearing a plaid shirt, working in a studio or workshop. He is focused on an object on a white surface in front of him.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-20">
      <a class="sample-nav prev" href="#it2t-sample-19" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/69.png' | relative_url }}" alt="图像条件文本生成样本 20" />
      <a class="sample-nav next" href="#it2t-sample-21" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image shows a newlywed couple at a wedding, dressed in wedding attire.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-21">
      <a class="sample-nav prev" href="#it2t-sample-20" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/70.png' | relative_url }}" alt="图像条件文本生成样本 21" />
      <a class="sample-nav next" href="#it2t-sample-22" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image is a cartoon-style drawing of a character dressed in a colored outfit, with a body showing green and yellow stripes. The character is touching its face with one hand.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-22">
      <a class="sample-nav prev" href="#it2t-sample-21" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/26.png' | relative_url }}" alt="图像条件文本生成样本 22" />
      <a class="sample-nav next" href="#it2t-sample-23" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows musicians performing with instruments at a ceremony. The background includes a floor and a wall.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-23">
      <a class="sample-nav prev" href="#it2t-sample-22" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/74.png' | relative_url }}" alt="图像条件文本生成样本 23" />
      <a class="sample-nav next" href="#it2t-sample-24" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows a black sleeveless T-shirt with a printed design on the front.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-24">
      <a class="sample-nav prev" href="#it2t-sample-23" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/75.png' | relative_url }}" alt="图像条件文本生成样本 24" />
      <a class="sample-nav next" href="#it2t-sample-25" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows a person wearing a white blouse over a black and white striped skirt.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-25">
      <a class="sample-nav prev" href="#it2t-sample-24" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/78.png' | relative_url }}" alt="图像条件文本生成样本 25" />
      <a class="sample-nav next" href="#it2t-sample-26" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows a white Toyota car with a sleek design, featuring prominent front and side headlights.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-26">
      <a class="sample-nav prev" href="#it2t-sample-25" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/79.png' | relative_url }}" alt="图像条件文本生成样本 26" />
      <a class="sample-nav next" href="#it2t-sample-27" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows a man wearing a black baseball cap and a grey shirt. The focus is on the cap, and the background is a plain, neutral grey.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-27">
      <a class="sample-nav prev" href="#it2t-sample-26" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/80.png' | relative_url }}" alt="图像条件文本生成样本 27" />
      <a class="sample-nav next" href="#it2t-sample-28" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows a necklace with a red, ball-shaped pendant.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-28">
      <a class="sample-nav prev" href="#it2t-sample-27" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/81.png' | relative_url }}" alt="图像条件文本生成样本 28" />
      <a class="sample-nav next" href="#it2t-sample-29" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows a person standing on a paved area with a scenic green backdrop.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-29">
      <a class="sample-nav prev" href="#it2t-sample-28" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/90.png' | relative_url }}" alt="图像条件文本生成样本 29" />
      <a class="sample-nav next" href="#it2t-sample-30" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows an old building with distinctive traditional Chinese-style architecture. It features traditional entrance details, and an iron structure with a roof is visible above the building.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-30">
      <a class="sample-nav prev" href="#it2t-sample-29" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/91.png' | relative_url }}" alt="图像条件文本生成样本 30" />
      <a class="sample-nav next" href="#it2t-sample-31" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows a decorative ornament in the shape of a flower.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-31">
      <a class="sample-nav prev" href="#it2t-sample-30" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/94.png' | relative_url }}" alt="图像条件文本生成样本 31" />
      <a class="sample-nav next" href="#it2t-sample-32" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows several people playing volleyball, with one of them jumping up. The scene takes place under a clear afternoon sky.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-32">
      <a class="sample-nav prev" href="#it2t-sample-31" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/48.png' | relative_url }}" alt="图像条件文本生成样本 32" />
      <a class="sample-nav next" href="#it2t-sample-33" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Generate a detailed caption for this image.<br /><strong>Output:</strong> The image shows a soccer player in action during a match, wearing a numbered uniform. The player appears to be running and kicking the ball on the field.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-33">
      <a class="sample-nav prev" href="#it2t-sample-32" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/50.png' | relative_url }}" alt="图像条件文本生成样本 33" />
      <a class="sample-nav next" href="#it2t-sample-34" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Generate a detailed caption for this image.<br /><strong>Output:</strong> The image shows a black T-shirt with a design featuring a blue cat printed on the front.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-34">
      <a class="sample-nav prev" href="#it2t-sample-33" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/51.png' | relative_url }}" alt="图像条件文本生成样本 34" />
      <a class="sample-nav next" href="#it2t-sample-35" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Generate a detailed caption for this image.<br /><strong>Output:</strong> The image shows a young boy wearing a grey vest over a white shirt, with a blue accent.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-35">
      <a class="sample-nav prev" href="#it2t-sample-34" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/96.png' | relative_url }}" alt="图像条件文本生成样本 35" />
      <a class="sample-nav next" href="#it2t-sample-36" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Generate a detailed caption for this image.<br /><strong>Output:</strong> The image shows a young girl with straight hair, wearing a yellow crop top and white shorts. She is making a fist with one hand and jumping, set against a blue background.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-36">
      <a class="sample-nav prev" href="#it2t-sample-35" aria-label="上一张">‹</a>
      <img src="{{ '/assets/it2t_samples/97.png' | relative_url }}" alt="图像条件文本生成样本 36" />
      <a class="sample-nav next" href="#it2t-sample-1" aria-label="下一张">›</a>
      <figcaption><strong>Prompt:</strong> Generate a detailed caption for this image.<br /><strong>Output:</strong> The image shows the stands of a large soccer stadium. The field and running track are visible, with green grass surrounding the stands and a few spectators present.</figcaption>
    </figure>
  </div>
  <div class="sample-dots" aria-label="图像条件文本生成样本快速跳转">
    <a href="#it2t-sample-1">1</a>
    <a href="#it2t-sample-2">2</a>
    <a href="#it2t-sample-3">3</a>
    <a href="#it2t-sample-4">4</a>
    <a href="#it2t-sample-5">5</a>
    <a href="#it2t-sample-6">6</a>
    <a href="#it2t-sample-7">7</a>
    <a href="#it2t-sample-8">8</a>
    <a href="#it2t-sample-9">9</a>
    <a href="#it2t-sample-10">10</a>
    <a href="#it2t-sample-11">11</a>
    <a href="#it2t-sample-12">12</a>
    <a href="#it2t-sample-13">13</a>
    <a href="#it2t-sample-14">14</a>
    <a href="#it2t-sample-15">15</a>
    <a href="#it2t-sample-16">16</a>
    <a href="#it2t-sample-17">17</a>
    <a href="#it2t-sample-18">18</a>
    <a href="#it2t-sample-19">19</a>
    <a href="#it2t-sample-20">20</a>
    <a href="#it2t-sample-21">21</a>
    <a href="#it2t-sample-22">22</a>
    <a href="#it2t-sample-23">23</a>
    <a href="#it2t-sample-24">24</a>
    <a href="#it2t-sample-25">25</a>
    <a href="#it2t-sample-26">26</a>
    <a href="#it2t-sample-27">27</a>
    <a href="#it2t-sample-28">28</a>
    <a href="#it2t-sample-29">29</a>
    <a href="#it2t-sample-30">30</a>
    <a href="#it2t-sample-31">31</a>
    <a href="#it2t-sample-32">32</a>
    <a href="#it2t-sample-33">33</a>
    <a href="#it2t-sample-34">34</a>
    <a href="#it2t-sample-35">35</a>
    <a href="#it2t-sample-36">36</a>
  </div>
</div>

> *关于图像说明。* 图像×文本→文本示例中的图像由**外部图像生成模型根据真值描述生成**，以避免真实照片相关的版权问题。

### 4.3 文本 → 文本 {#text-to-text}

尽管仅使用约 **10 亿** 文本 token 进行训练，统一模型在对话、叙事、说明、技术与文章式提示下仍能保持较为连贯的文本续写能力。

---

## 5. 后续实验：从可行性到定量对比 {#future-experiments}

本文当前结果主要验证了该统一建模方案的**可行性**：单一模型可在同一预训练配方下同时获得文本生成、图像生成与初步图文理解能力。后续工作需要进一步回答该统一模型在放大预训练规模并引入必要的 SFT/RL 后训练后，是否优于标准替代方案。为此，后续我们将开展以下两类定量对照实验。

**(Q1) 在受控设置下，统一多模态预训练是否改善生成（pixel output）？**
首先，本文将与 **标准 MMDiT** 基线进行直接比较。该基线采用常规文本→图像设置，其中文本仅作为条件输入，而不参与联合潜变量建模。比较将在**数据、参数量与训练样本数匹配**的条件下进行，以隔离训练范式本身的影响，并检验共享 MMDiT 建模联合潜变量分布是否能带来可测的生成质量增益。

**(Q2) 统一多模态预训练是否也改善理解（text output）？**
其次，除生成质量外，后续还将评估共享 MMDiT 建模是否能改善多模态理解能力，并与其他统一建模范式进行系统比较。

此外，后续研究将报告 scaling 行为、受控匹配对照以及标准生成与理解基准结果。

---

## 6. 路线图：共享联合先验下的更多模态 {#roadmap}

本框架可以自然扩展至更多模态，并配以相应的块布局与注意力规则。可扩展的模态包括：

- **视频。** 时间分块的潜变量可自然适配块因果结构：每帧或片段构成一个块，块间采用时间因果结构，块内采用双向建模。这一形式支持在单一序列内进行图像、文本与视频的交错生成和理解。
- **音频。** 连续音频潜变量可作为另一类段类型，用于建模文本↔音频、视频↔音频之间的对齐关系，例如描述、旁白和声音条件生成。
- **动作。** 连续动作潜变量（轨迹、控制信号）可被纳入共享块因果 MMDiT 中，从而支持感知与动作的联合推理，并为世界模型式建模提供接口。
<!-- - **中间推理潜变量。** 由于文本与其他模态均以因果块形式组织，模型原则上可在生成图像、视频或动作前生成中间推理潜变量，为面向多模态生成的显式推理路径提供结构支持。 -->

---


## References {#references}

[1] H. Guo, Q. Zhao, Y. Zhao, S. Nie, R. Zhu, Q. Guo, F. Wang, T. Yang, H. Zhao, G. Wei, and Y. Zeng, "Continuous Latent Diffusion Language Model," arXiv:2605.06548, 2026. <https://arxiv.org/abs/2605.06548>

[2] Chameleon Team, "Chameleon: Mixed-Modal Early-Fusion Foundation Models," arXiv:2405.09818, 2024. <https://arxiv.org/abs/2405.09818>

[3] C. Wu et al., "Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation," arXiv:2410.13848, 2024. <https://arxiv.org/abs/2410.13848>

[4] X. Chen, Z. Wu, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, and C. Ruan, "Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling," arXiv:2501.17811, 2025. <https://arxiv.org/abs/2501.17811>

[5] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, "Sigmoid Loss for Language Image Pre-Training," ICCV, 2023. <https://arxiv.org/abs/2303.15343>

[6] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural Discrete Representation Learning," NeurIPS, 2017. <https://arxiv.org/abs/1711.00937>

[7] C. Zhou, L. Yu, A. Babu, K. Tirumala, M. Yasunaga, L. Shamis, J. Kahn, X. Ma, L. Zettlemoyer, and O. Levy, "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model," arXiv:2408.11039, 2024. <https://arxiv.org/abs/2408.11039>

[8] C. Deng et al., "Emerging Properties in Unified Multimodal Pretraining," arXiv:2505.14683, 2025. <https://arxiv.org/abs/2505.14683>

[9] J. Xie et al., "Show-o: One Single Transformer to Unify Multimodal Understanding and Generation," arXiv:2408.12528, 2024. <https://arxiv.org/abs/2408.12528>

[10] J. Xie, Z. Yang, and M. Z. Shou, "Show-o2: Improved Native Unified Multimodal Models," arXiv:2506.15564, 2025. <https://arxiv.org/abs/2506.15564>

[11] Z. Liu et al., "Tuna: Taming Unified Visual Representations for Native Unified Multimodal Models," arXiv:2512.02014, 2025. <https://arxiv.org/abs/2512.02014>

[12] Z. Liu et al., "Tuna-2: Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation," arXiv:2604.24763, 2026. <https://arxiv.org/abs/2604.24763>

[13] X. Pan et al., "Transfer between Modalities with MetaQueries," arXiv:2504.06256, 2025. <https://arxiv.org/abs/2504.06256>

[14] C. Wu et al., "Qwen-Image Technical Report," arXiv:2508.02324, 2025. <https://arxiv.org/abs/2508.02324>

[15] B. Lin et al., "UniWorld-V1: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation," arXiv:2506.03147, 2025. <https://arxiv.org/abs/2506.03147>

[16] C. Wu et al., "OmniGen2: Towards Instruction-Aligned Multimodal Generation," arXiv:2506.18871, 2025. <https://arxiv.org/abs/2506.18871>

[17] C. Wei et al., "UniVideo: Unified Understanding, Generation, and Editing for Videos," arXiv:2510.08377, 2025. <https://arxiv.org/abs/2510.08377>

[18] W. Peebles and S. Xie, "Scalable Diffusion Models with Transformers," arXiv:2212.09748, 2022. <https://arxiv.org/abs/2212.09748>

[19] P. Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis," ICML, 2024. <https://arxiv.org/abs/2403.03206>

[20] S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and S. Xie, "Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think," ICLR, 2025. <https://arxiv.org/abs/2410.06940>

Cola DLM 项目页：<https://hongcanguo.github.io/Cola-DLM/>
