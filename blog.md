---
layout: default
lang: en
title: "Unified Multimodal Flow Matching with Cola DLM"
thumbnail: /assets/fig-unified-overview.png
authors: Yian Zhao
permalink: /blog/2026/unified-cola-en/
tags:
  - unified multimodal pretraining
  - latent diffusion model
  - flow matching
---
**English** | [中文]({{ '/blog/2026/unified-cola/' | relative_url }})

# Unified Multimodal Flow Matching with Cola DLM

<div class="post-meta">
  <span>Published: 2026-06-09</span>
  <span>Words: about 5,500</span>
  <span>Reading time: about 25 min</span>
  <span id="busuanzi_container_page_pv">Views: <span id="busuanzi_value_page_pv">Loading</span></span>
</div>

<details class="toc-card" open markdown="1">
<summary>Contents</summary>

- [Abstract](#abstract)
- [1. Motivation: Why a Different Route to Unified Modeling?](#motivation)
  - [A Brief Recap of Cola DLM](#cola-dlm-recap)
- [2. A Joint Distribution over Continuous Latents](#joint-latents)
- [3. Architecture and Training](#architecture-training)
  - [Sequence Layout](#sequence-layout)
  - [Attention Masks as Task Semantics](#attention-masks)
  - [Objective](#objective)
  - [Training Setup](#training-setup)
- [4. Qualitative Results](#qualitative-results)
  - [4.1 Text-to-image](#text-to-image)
  - [4.2 Image-and-text-to-text](#image-text-to-text)
  - [4.3 Text-to-text](#text-to-text)
- [5. Future Experiments: From Feasibility to Quantitative Comparison](#future-experiments)
- [6. Roadmap: More Modalities under a Shared Joint Prior](#roadmap)
- [References](#references)
</details>

<div class="post-note" markdown="1">
**Reading guide.** The post starts with motivation and joint latent modeling, then moves into training design and qualitative results; for a quick pass, start with Sections 1, 3, and 4.
</div>

### Abstract {#abstract}
> Recent unified multimodal pretraining has moved beyond early single-stream autoregressive sequence modeling toward two-tower Reasoner--Generator paradigms, including cascaded MLLM-to-diffusion systems and parallel Mixture-of-Transformer (MoT) architectures. This post explores a different route built on **Cola DLM** (Continuous Latent Diffusion Language Model) [1]. We map both text and visual signals into continuous latent spaces and use a shared block-causal MMDiT to parameterize a latent generative distribution, so that understanding (text output) and generation (pixel output) can be trained through a unified interface. From this perspective, understanding and generation are different conditional views of the same multimodal joint distribution. Joint multitask pretraining constrains the shared generative distribution from multiple conditional directions, encouraging the model to learn semantic representations, cross-modal alignment, and generative dynamics in one representation space. We describe the architecture, its main design choices, and qualitative results on **text-to-text**, **text-to-image**, and **image-and-text-to-text** pretraining tasks.
<!-- The current experiments are positioned as a proof of concept: we validate the feasibility and early convergence behavior of the unified architecture, without large-scale SFT or RL post-training. -->

---

## 1. Motivation: Why a Different Route to Unified Modeling? {#motivation}

In recent years, the central question in unified multimodal pretraining has shifted from whether a single system can support both understanding and generation to how the understanding and generation pathways should interact. At the architectural level, existing approaches can be broadly grouped into three lines of work.

**Route 1: unified autoregressive sequence modeling.** Chameleon [2] discretizes both text and images into tokens and predicts them with a single Transformer. Janus and Janus-Pro [3,4] further decouple the visual understanding encoder, such as SigLIP [5], from the discrete image tokenizer, such as VQ-VAE [6], while the generation path still largely relies on autoregressive modeling over discrete visual tokens. This route is closely aligned with the language-modeling paradigm and naturally supports interleaved inputs and outputs in a unified sequence format. Its limitation is that image generation quality and sampling efficiency are often constrained by discrete tokenization and long autoregressive sequences.

**Route 2: parallel Reasoner--Generator interaction.** Transfusion [7] jointly trains next-token prediction for text and diffusion objectives for images in a single multimodal model. BAGEL [8] adopts an MoT architecture in which understanding and generation experts operate over the same multimodal sequence and interact through shared self-attention. Show-o and Show-o2 [9,10] further study how to combine autoregressive text modeling and discrete diffusion-based visual generation inside a single Transformer. Tuna and Tuna-2 [11,12] start from unified visual representations and explore end-to-end unified modeling with continuous visual representations or pixel embeddings. These methods preserve autoregressive modeling on the text side while introducing diffusion objectives for the visual side, making them better suited to high-fidelity image generation than purely autoregressive systems. At the same time, they must handle more complex optimization coupling among representation learning, task mixture, and generation/understanding objectives.

**Route 3: cascaded Reasoner-to-Generator bridging.** MetaQueries [13] uses learnable queries to extract generation conditions from a pretrained MLLM and feeds them through a connector into an MMDiT. Qwen-Image [14] uses Qwen2.5-VL as the semantic conditioner and combines it with a VAE and an MMDiT for image generation and editing. UniWorld-V1 [15], OmniGen2 [16], and UniVideo [17] follow similar ideas, transferring the understanding capability of an MLLM into conditioning signals for an MMDiT and extending the framework to image editing, in-context generation, and video generation/editing. This route has strong engineering scalability because it can reuse existing MLLM and MMDiT backbones. However, understanding and generation are usually connected through hidden states, queries, connectors, or dual-stream conditioning. The unification is therefore more system-level composition than direct modeling of a multimodal joint distribution by a single model.

These lines of work suggest that "unification" should not simply mean sharing one Transformer, nor should it be reduced to attaching a diffusion generator after a language model. Even when visual generation happens in a VAE latent space, cross-modal learning in many existing systems remains closer to conditional modeling: one modality conditions another, rather than the model explicitly capturing a joint distribution over multimodal latents.

Motivated by this observation, we aim to construct a unified training paradigm distinct from both parallel two-tower interaction and cascaded bridging. Text and visual signals are mapped by VAEs into continuous latent spaces, and a single shared block-causal MMDiT [18,19] models

$$
p_\psi\big(z_0^{\text{text}}, z_0^{\text{pixel}}\big).
$$

Under this view, understanding (text output) and generation (pixel output) correspond to different conditional views of the same multimodal joint distribution. Multitask training constrains $p_\psi(z_0^{\text{text}}, z_0^{\text{pixel}})$ from multiple conditional directions, encouraging the model to capture mutual information between $Z^{\text{text}}$ and $Z^{\text{pixel}}$ and to share semantic representation, cross-modal alignment, and generative dynamics under one interface.

Cola DLM provides a mature continuous-latent formulation for text generation. In this post, we show how the same formulation can be extended naturally to other modalities and used as the basis for this unified training paradigm.

### A Brief Recap of Cola DLM {#cola-dlm-recap}

Cola DLM is a continuous latent-variable diffusion language model. Its key idea is to avoid denoising at the token level. Instead, it:

1. learns a stable **text-to-continuous-latent** and **continuous-latent-to-text** mapping with a **Text VAE**;
2. models a **global semantic prior** in continuous latent space with a **block-causal DiT**;
3. generates text in a streaming manner through a **conditional decoder**.

Diffusion is used for **latent prior transport**:

$$
z_1 \sim p_1,\qquad z_0 = \Phi^{\psi}_{0\leftarrow 1}(z_1),\qquad x \sim p_\theta(x \mid z_0).
$$

This decomposition explicitly separates **global semantic organization** in continuous latent space from **local textual realization** in the decoder. The same separation becomes crucial when extending the model to additional modalities.

---

## 2. A Joint Distribution over Continuous Latents {#joint-latents}

Unified modeling should not rely only on a shared backbone, nor should it force different modalities into exactly the same representation space. A more natural approach is to first map text and visual observations into continuous latent variables, and then let a shared block-causal MMDiT model the joint distribution over these latents. In this way, textual semantics, visual content, and their correspondence are all handled through the same interface.

Concretely, we follow the same probabilistic decomposition as Cola DLM. Let $x_{\text{text}}$ and $x_{\text{pixel}}$ denote text and visual observations. Their corresponding encoders produce continuous latents:

$$
z_0^{\text{text}} \in \mathcal{Z}_{\text{text}},\quad
z_0^{\text{text}} \sim q_{\phi_{\text{text}}}(z \mid x_{\text{text}}),
\qquad
z_0^{\text{pixel}} \in \mathcal{Z}_{\text{pixel}},\quad
z_0^{\text{pixel}} \sim q_{\phi_{\text{pixel}}}(z \mid x_{\text{pixel}}).
$$

We then form a **joint latent variable** and model a unified generative process:

$$
\bar{z}_0 = \big(z_0^{\text{text}},\, z_0^{\text{pixel}}\big),
\qquad
p(x_{\text{text}}, x_{\text{pixel}}, \bar{z}_0) = p_\theta\big(x_{\text{text}}, x_{\text{pixel}} \mid \bar{z}_0\big)\, p_\psi(\bar{z}_0).
$$

The **modality-specific VAE encoders and decoders** convert between observation spaces and latent spaces, while the **shared block-causal MMDiT** parameterizes $p_\psi(\bar z_0)$ over $\mathcal{Z}_{\text{text}}\times\mathcal{Z}_{\text{pixel}}$, learning the joint distribution of text and visual latents. Prior transport now acts on the multimodal latent variable $\bar{z}_0$: the shared block-causal MMDiT first generates the joint latent state, and the corresponding decoders then produce text or pixel outputs. Continuous latents provide a unified modeling interface; they do not require all modalities to share an identical representation.

From the joint-distribution perspective, the training objective can be written as

$$
\mathbb{E}[\mathcal{L}_{\text{ELBO}}]
:= \mathbb{E}_q\big[\log p_\theta(x_{\text{text}}, x_{\text{pixel}} \mid \bar{z}_0)\big]
- I\big((X_{\text{text}}, X_{\text{pixel}});\, \bar{Z}_0\big)
- \mathrm{KL}\big(\bar{q}(\bar{z}_0)\,\|\,p_\psi(\bar{z}_0)\big).
$$

The three terms correspond to three basic questions: whether the decoders can reconstruct text and visual observations from the joint latent $\bar z_0$; how much information about the input is retained in the latent variable; and whether the aggregated posterior $\bar q(\bar z_0)$ produced by the encoders can be matched by the shared prior $p_\psi(\bar z_0)$.

In this framework, text latents and visual latents need not be forced into pointwise comparable representations. The model can learn different conditional generation directions within the same joint distribution, such as $p_\psi(z^{\text{pixel}}\mid z^{\text{text}})$ or $p_\psi(z^{\text{text}}\mid z^{\text{pixel}})$.

Thus, the latent variables carry compressed high-level semantics, while the decoders realize those semantics as concrete text or pixel outputs. The unification happens at the level of the joint distribution over continuous latents, not merely through the reuse of a cross-modal backbone.

---

## 3. Architecture and Training {#architecture-training}

![**Figure 1.** Unified text--vision modeling with Cola DLM. Left: text continuation and image-conditioned text generation. Middle: text-to-image generation. Right: method overview. Text and visual signals are mapped into continuous latents and modeled by a shared block-causal MMDiT as a joint generative distribution.]({{ '/assets/fig-unified-overview.png' | relative_url }})

The overall architecture is shown on the right of Figure 1. For text and images, it contains the following components:

- **Text path.** A **Text VAE** maps text into continuous latents, and the text sequence is **partitioned into blocks**.
- **Image path.** An **Image VAE** maps images into compact latents, with each image treated as **a single block**.
- **Shared block-causal MMDiT.** This module operates over both text blocks and image blocks, parameterizes the joint latent generative distribution, and supports both **intra-modal** processing and **cross-modal** interaction. In the current image model, the stack uses roughly **one third dual-stream** and **two thirds single-stream** DiT blocks.

A single model supports three tasks under the same framework:

- **Text-to-text (T2T)**,
- **Text-to-image (T2I)**,
- **Image-and-text-to-text (IT2T)**, including captioning and visual question answering.

### Sequence Layout {#sequence-layout}

We pack continuous latents into a sequence and annotate each position with two integers. Suppose a packed sequence has $N$ positions. Position $i$ carries:

- a **segment type** $s_i \in \{\textsf{P}, \textsf{C}, \textsf{I}, \textsf{N}\}$, denoting *instruction text*, *clean text*, *image*, and *noisy text*, respectively;
- a **block index** $b_i \in \mathbb{Z}$, which partitions text into contiguous blocks of size $B$ (we use $B=16$ in the current implementation).

Text latents are split by block size $B$, while an image latent is treated as a *single* block. These two annotations carry all task semantics:

- **Clean text** ($\textsf{C}$) latents are **context/conditioning** and receive **no loss**.
- **Noisy text** ($\textsf{N}$) latents are **generation targets** and receive the diffusion loss.
- **Instruction** ($\textsf{P}$, used only for IT2T) latents are task **prompts** and also serve as context. They are separated from $\textsf{C}$ to avoid mixing the condition with the target, as explained in the IT2T section below.
- **Block indices** impose **causal order** across text blocks while keeping each block, and each single image block, **bidirectional** internally.

The task is defined only by (i) which segments are present and in what order, and (ii) the attention rule that couples them. The layouts are:

| Task | Per-sample segment layout | Diffusion target |
| --- | --- | --- |
| Text-to-text (T2T) | $[\,\textsf{C}\,]\,[\,\textsf{N}\,]$ | text blocks |
| Text-to-image (T2I) | $[\,\textsf{C}\,]\,[\,\textsf{I}\,]\,[\,\textsf{N}\,]$ | image (primary) + caption (auxiliary) |
| Image-and-text-to-text (IT2T) | $[\,\textsf{P}\,]\,[\,\textsf{I}\,]\,[\,\textsf{C}\,]\,[\,\textsf{N}\,]$ | text only ($\textsf{P},\textsf{I}$ are fixed context) |

Within a single forward pass, multiple samples, possibly from different tasks under gradient accumulation, are concatenated into one long sequence. Attention is additionally constrained to stay **within the same sample**: query position $i$ may attend only to key position $j$ from the same sample.

### Attention Masks as Task Semantics {#attention-masks}

The task semantics of the shared block-causal MMDiT are controlled by the attention mask $M_{ij}\in\{0,1\}$, which determines whether query $i$ may attend to key $j$. The mask is defined by the following segment-conditioned rules. **Figure 2** visualizes small rendered examples for each task, with rows as queries and columns as attended keys.

![**Figure 2.** Attention masks of the shared block-causal MMDiT under the three tasks, rendered from the segment-conditioned rules below. Rows are queries and columns are attended keys. Red indicates allowed attention and gray indicates masked positions. The color strips along the top and left encode segment type (green = IT2T instruction segment $\textsf{P}$, blue = clean text, orange = image, purple = noisy text).]({{ '/assets/fig-attention-masks.png' | relative_url }})

**Text-to-image.** Image generation is conditioned on clean text. We additionally append noisy text to enable multitask training inside a single sequence (T2I + T2T). The attention rule is:

$$
M_{ij}=1 \iff
\begin{cases}
s_j=\textsf{C}\ \wedge\ b_j\le b_i, & s_i=\textsf{C}\quad(\text{block-causal caption}),\\[2pt]
s_j\in\{\textsf{C},\textsf{I}\}, & s_i=\textsf{I}\quad(\text{full caption conditioning + bidirectional self-attention}),\\[2pt]
(s_j=\textsf{C}\ \wedge\ b_j<b_i)\ \vee\ (s_j=\textsf{N}\ \wedge\ b_j=b_i), & s_i=\textsf{N}\quad(\text{block-diffusion text}).
\end{cases}
$$

The image segment and noisy-text segment do **not** attend to each other. Therefore, the two objectives are not coupled through the auxiliary text loss directly. They are coupled through (i) the image attending to the clean text, so that the text condition receives gradients from the image loss, and (ii) the shared block-causal MMDiT parameters.

**Image-and-text-to-text.** Both the **instruction** $\textsf{P}$ and the **image** $\textsf{I}$ are clean conditions, and the model generates a textual answer conditioned on them. The key design issue is that the clean text stream in IT2T contains both the task prompt and a teacher-forcing copy of the answer. If they are merged into the same segment, the answer can leak into the image representation and then be read by the noisy text, short-circuiting the diffusion objective. If the image cannot see the prompt at all, the visual encoding becomes task-agnostic. We therefore put the instruction in a separate segment $\textsf{P}$ and keep it separate from the answer context $\textsf{C}$:

$$
M_{ij}=1 \iff
\begin{cases}
s_j\in\{\textsf{P},\textsf{I}\}, & s_i=\textsf{P}\quad(\text{instruction and image form a bidirectional condition}),\\[2pt]
s_j\in\{\textsf{P},\textsf{I}\}, & s_i=\textsf{I}\quad(\text{image sees the instruction, but not the answer}),\\[2pt]
(s_j=\textsf{C}\ \wedge\ b_j\le b_i)\ \vee\ s_j\in\{\textsf{P},\textsf{I}\}, & s_i=\textsf{C}\quad(\text{answer context conditioned on prompt and image}),\\[2pt]
(s_j=\textsf{C}\ \wedge\ b_j<b_i)\ \vee\ (s_j=\textsf{N}\ \wedge\ b_j=b_i)\ \vee\ s_j\in\{\textsf{P},\textsf{I}\}, & s_i=\textsf{N}\quad(\text{block-diffusion answer conditioned on prompt and image}).
\end{cases}
$$

The instruction and image thus form a single fully bidirectional condition block, allowing the visual encoding to be prompt-aware. At the same time, neither attends to the answer ($\textsf{C}$ or $\textsf{N}$), preventing the answer from leaking backward into the condition. The answer part, consisting of the $\textsf{C}$ context and $\textsf{N}$ target, follows the block-causal structure and is conditioned on $\{\textsf{P},\textsf{I}\}$. Because the image is never used as an answer query target in this task, it serves only as context and the image diffusion loss is disabled.

**Text-to-text** is the special case of the T2I rule with the image segment removed: a clean prefix is encoded in a block-causal manner, and the remaining text blocks are generated by block diffusion conditioned on previous clean blocks.

### Objective {#objective}

The per-step loss is the sum of all active objectives:

$$
\mathcal{L} = \mathcal{L}_{\text{image}} \;+\; \mathcal{L}_{\text{text}} \;+\; \mathcal{L}_{\text{REPA}},
$$

Each term is a velocity-prediction MSE in the corresponding latent space. $\mathcal{L}_{\text{REPA}}$ is an optional representation-alignment term [20], active only for T2I, that aligns intermediate DiT features with visual-encoder features to accelerate generation convergence. During training, tasks are sampled according to configurable task ratios, and the task choice is synchronized across workers to keep distributed collective calls consistent.

### Training Setup {#training-setup}

- **Noise schedule.** Both modalities use a rectified-flow linear interpolation schedule, velocity prediction, and an Euler sampler.
- **Block-level noise.** For text, each block samples its own timestep and all tokens within the block share that timestep, following the hierarchical block-causal design of Cola DLM. Image latents are noised as a single block.
- **Modality-specific timestep sampling.** Training timesteps are sampled independently for image and text blocks from logit-normal distributions.
- **Classifier-free guidance.** During training, the entire text condition of a T2I sample is randomly dropped so that the model learns an unconditional visual generation distribution.

All modules are **pretrained from scratch**, with task data ratios adjusted across pretraining stages. The results reported here correspond to a proof-of-concept pretraining scale, without large-scale instruction tuning or RL post-training:

| Task | Setup |
| --- | --- |
| Text-to-image | 256 resolution for **80k** steps (global batch ≈ **3k**), followed by 640 resolution for **10k** steps (global batch ≈ **1k**) |
| Image-and-text-to-text | same batch configuration, about **50k** steps |
| Text data (total) | about **1B** text tokens |
| Multimodal supervision | about **5M** image-text pairs |

Under this limited data and compute budget, the model can already generate coherent text, produce reasonably structured images, and show preliminary image-captioning ability. This suggests that learning a joint distribution first in latent space and then delegating final realization to modality-specific decoders may be a data-efficient route to cross-modal capability.

---

## 4. Qualitative Results {#qualitative-results}

This section reports qualitative results on the three tasks.

### 4.1 Text-to-image {#text-to-image}

Click any thumbnail to inspect the image in a larger viewer and continue browsing sample by sample.


<div class="sample-carousel" aria-label="Text-to-image sample gallery">
  <div class="sample-track">
    <figure class="sample-slide" id="t2i-sample-1">
      <a class="sample-nav prev" href="#t2i-sample-39" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/56.jpg' | relative_url }}" alt="Text-to-image sample 1" />
      <a class="sample-nav next" href="#t2i-sample-2" aria-label="Next sample">›</a>
      <figcaption>A delicate white dandelion seed head stands above a green meadow, with other soft, blurred flowers nearby. A dark row of trees and a cloudy blue-gray sky give the open field a calm, breezy spring atmosphere.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-2">
      <a class="sample-nav prev" href="#t2i-sample-1" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/82.jpg' | relative_url }}" alt="Text-to-image sample 2" />
      <a class="sample-nav next" href="#t2i-sample-3" aria-label="Next sample">›</a>
      <figcaption>Two large abstract paintings dominate a polished interior hallway, their vivid colors contrasting with the neutral wall panels and glossy floor. Ceiling spotlights, patterned rugs, and nearby doorways create the feeling of a private gallery or elegant office space.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-3">
      <a class="sample-nav prev" href="#t2i-sample-2" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/103.jpg' | relative_url }}" alt="Text-to-image sample 3" />
      <a class="sample-nav next" href="#t2i-sample-4" aria-label="Next sample">›</a>
      <figcaption>Several boats glide across rippling water during sunset, appearing as dark silhouettes beneath a long arched bridge. The orange sun and its reflection stretch across the river, giving the scene a warm, tranquil waterfront mood.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-4">
      <a class="sample-nav prev" href="#t2i-sample-3" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/123.jpg' | relative_url }}" alt="Text-to-image sample 4" />
      <a class="sample-nav next" href="#t2i-sample-5" aria-label="Next sample">›</a>
      <figcaption>White spring blossoms cluster thickly along interwoven tree branches, mixed with small green leaves and unopened buds. The shallow depth of field makes the foreground flowers stand out while the background turns into a delicate canopy of pale bloom.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-5">
      <a class="sample-nav prev" href="#t2i-sample-4" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/134.jpg' | relative_url }}" alt="Text-to-image sample 5" />
      <a class="sample-nav next" href="#t2i-sample-6" aria-label="Next sample">›</a>
      <figcaption>Golden sunset light spreads behind a line of dark mountain silhouettes and reflects across the still lake below. Grasses in the foreground add depth, while the mirrored peaks and warm sky create a peaceful landscape composition.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-6">
      <a class="sample-nav prev" href="#t2i-sample-5" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/280.jpg' | relative_url }}" alt="Text-to-image sample 6" />
      <a class="sample-nav next" href="#t2i-sample-7" aria-label="Next sample">›</a>
      <figcaption>A broad blue lake stretches toward layered mountains beneath a sweeping sky of textured white clouds. The calm ripples, low dark shorelines, and distant ridges create a cool, expansive landscape with a quiet sense of depth.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-7">
      <a class="sample-nav prev" href="#t2i-sample-6" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/h1.jpg' | relative_url }}" alt="Text-to-image sample 7" />
      <a class="sample-nav next" href="#t2i-sample-8" aria-label="Next sample">›</a>
      <figcaption>A young woman in a white coat and patterned scarf stands at night in front of glowing red and yellow lanterns. The warm lights and wooden structures behind her create a festive, atmospheric portrait with soft contrast.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-8">
      <a class="sample-nav prev" href="#t2i-sample-7" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/l1.jpg' | relative_url }}" alt="Text-to-image sample 8" />
      <a class="sample-nav next" href="#t2i-sample-9" aria-label="Next sample">›</a>
      <figcaption>A monochrome mountain landscape drawing shows steep ridges, deep valleys, winding water, and layered terrain fading into the distance. The sketch-like shading gives the scene a rugged, hand-rendered quality, as if from a travel notebook.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-9">
      <a class="sample-nav prev" href="#t2i-sample-8" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/67.jpg' | relative_url }}" alt="Text-to-image sample 9" />
      <a class="sample-nav next" href="#t2i-sample-10" aria-label="Next sample">›</a>
      <figcaption>Tall trees on both sides frame a dramatic sunset sky filled with glowing orange, pale yellow, and blue cloud textures. The dark silhouettes below make the scene feel quiet, spacious, and gently illuminated by evening light.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-10">
      <a class="sample-nav prev" href="#t2i-sample-9" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/83.jpg' | relative_url }}" alt="Text-to-image sample 10" />
      <a class="sample-nav next" href="#t2i-sample-11" aria-label="Next sample">›</a>
      <figcaption>A cozy outdoor dining area is lit by a large yellow umbrella and several red hanging lanterns. Wooden tables, simple chairs, and warm pools of light create an intimate evening setting with a festive, traditional atmosphere.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-11">
      <a class="sample-nav prev" href="#t2i-sample-10" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/86.jpg' | relative_url }}" alt="Text-to-image sample 11" />
      <a class="sample-nav next" href="#t2i-sample-12" aria-label="Next sample">›</a>
      <figcaption>A dense column of black and gray smoke rises from a rural property near small buildings and scattered trees. The surrounding dirt paths, fields, and pale sky make the sudden fire or burning event feel stark and unsettling.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-12">
      <a class="sample-nav prev" href="#t2i-sample-11" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/99.jpg' | relative_url }}" alt="Text-to-image sample 12" />
      <a class="sample-nav next" href="#t2i-sample-13" aria-label="Next sample">›</a>
      <figcaption>A vivid red rose fills the image in extreme close-up, revealing many layered petals curling toward a tight spiral at the center. The blurred green background keeps attention on the flower's saturated color and soft texture.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-13">
      <a class="sample-nav prev" href="#t2i-sample-12" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/275.jpg' | relative_url }}" alt="Text-to-image sample 13" />
      <a class="sample-nav next" href="#t2i-sample-14" aria-label="Next sample">›</a>
      <figcaption>A bearded man with curly hair sits indoors wearing a dark blazer and lavender shirt. Warm lamp light, stacks of documents, and framed artwork behind him suggest an office or study, giving the portrait a serious conversational tone.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-14">
      <a class="sample-nav prev" href="#t2i-sample-13" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/308.jpg' | relative_url }}" alt="Text-to-image sample 14" />
      <a class="sample-nav next" href="#t2i-sample-15" aria-label="Next sample">›</a>
      <figcaption>A sightseeing boat moves along a wide river in front of grand historic architecture, including ornate towers, stone facades, and decorative rooflines. Reflections shimmer on the water, emphasizing the scale and elegance of the riverside cityscape.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-15">
      <a class="sample-nav prev" href="#t2i-sample-14" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/323.png' | relative_url }}" alt="Text-to-image sample 15" />
      <a class="sample-nav next" href="#t2i-sample-16" aria-label="Next sample">›</a>
      <figcaption>A single leafless tree stands prominently in rolling green grassland, its intricate branches spreading against a bright sky. Distant mountains and soft clouds frame the isolated tree, giving the image a quiet, almost sculptural quality.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-16">
      <a class="sample-nav prev" href="#t2i-sample-15" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/344.jpg' | relative_url }}" alt="Text-to-image sample 16" />
      <a class="sample-nav next" href="#t2i-sample-17" aria-label="Next sample">›</a>
      <figcaption>A simple stylized illustration presents a red rose with green leaves, centered inside a rounded purple background with a thick black outline. The bold colors, clean shapes, and cartoon-like linework make the flower feel decorative and playful.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-17">
      <a class="sample-nav prev" href="#t2i-sample-16" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/411.jpg' | relative_url }}" alt="Text-to-image sample 17" />
      <a class="sample-nav next" href="#t2i-sample-18" aria-label="Next sample">›</a>
      <figcaption>Bright pink blossoms fill the foreground in dense clusters, their yellow centers catching the light. More branches and flowers blur behind them, creating a saturated spring scene that feels lively, sunny, and full of floral texture.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-18">
      <a class="sample-nav prev" href="#t2i-sample-17" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/417.jpg' | relative_url }}" alt="Text-to-image sample 18" />
      <a class="sample-nav next" href="#t2i-sample-19" aria-label="Next sample">›</a>
      <figcaption>A loose navy blue sweater hangs from a wooden hanger against a plain white background. The soft fabric, wide sleeves, and relaxed drape are emphasized by the simple product-style composition and uncluttered studio-like setting.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-19">
      <a class="sample-nav prev" href="#t2i-sample-18" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/429.jpg' | relative_url }}" alt="Text-to-image sample 19" />
      <a class="sample-nav next" href="#t2i-sample-20" aria-label="Next sample">›</a>
      <figcaption>A misty mountain rises beyond a quiet lake, its slopes softened by gray haze and overcast light. Reeds in the foreground frame the water, adding depth to the subdued scene and strengthening its calm, contemplative mood.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-20">
      <a class="sample-nav prev" href="#t2i-sample-19" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/580.jpg' | relative_url }}" alt="Text-to-image sample 20" />
      <a class="sample-nav next" href="#t2i-sample-21" aria-label="Next sample">›</a>
      <figcaption>A calm lake reflects a bright blue sky and the low green hills beyond the opposite shore. Reeds and leafy plants in the foreground create a natural frame, while distant buildings add a small human presence to the peaceful view.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-21">
      <a class="sample-nav prev" href="#t2i-sample-20" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/591.jpg' | relative_url }}" alt="Text-to-image sample 21" />
      <a class="sample-nav next" href="#t2i-sample-22" aria-label="Next sample">›</a>
      <figcaption>A roadside viewpoint looks out across blue ocean water toward distant islands and hazy mountains. Green shrubs, tall grasses, a utility pole, and a small signboard sit beside the path, giving the scenic overlook a casual travel feeling.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-22">
      <a class="sample-nav prev" href="#t2i-sample-21" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/641.jpg' | relative_url }}" alt="Text-to-image sample 22" />
      <a class="sample-nav next" href="#t2i-sample-23" aria-label="Next sample">›</a>
      <figcaption>A cup of foamy coffee sits beside fried pastries sprinkled with powdered sugar on brown paper. The colorful table covering, golden snacks, and creamy drink create a cheerful cafe or street-food scene focused on sweet comfort food.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-23">
      <a class="sample-nav prev" href="#t2i-sample-22" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/689.jpeg' | relative_url }}" alt="Text-to-image sample 23" />
      <a class="sample-nav next" href="#t2i-sample-24" aria-label="Next sample">›</a>
      <figcaption>Heavy gray storm clouds gather low over city buildings, power lines, and rooftops, filling most of the frame with dark, textured sky. The urban skyline beneath looks small and tense, suggesting an approaching downpour or dramatic weather change.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-24">
      <a class="sample-nav prev" href="#t2i-sample-23" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/696.jpg' | relative_url }}" alt="Text-to-image sample 24" />
      <a class="sample-nav next" href="#t2i-sample-25" aria-label="Next sample">›</a>
      <figcaption>A large steamed dumpling or bun rests on a round wooden plate, garnished with lettuce and small red tomatoes. The bright tabletop, folded cloth, and clean plating make the simple food presentation look fresh and carefully arranged.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-25">
      <a class="sample-nav prev" href="#t2i-sample-24" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/721.jpg' | relative_url }}" alt="Text-to-image sample 25" />
      <a class="sample-nav next" href="#t2i-sample-26" aria-label="Next sample">›</a>
      <figcaption>A sculptural bust shows a realistic human face emerging from a rough, crown-like form and textured reddish base. The plain wall behind it keeps focus on the unusual material contrast, facial detail, and handmade artistic expression.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-26">
      <a class="sample-nav prev" href="#t2i-sample-25" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/760.jpg' | relative_url }}" alt="Text-to-image sample 26" />
      <a class="sample-nav next" href="#t2i-sample-27" aria-label="Next sample">›</a>
      <figcaption>A round celebration cake is decorated with pink frosting roses, green leaves, and a large bow across the top. Plates of food around it suggest a shared meal, while the elaborate icing makes the cake the table's centerpiece.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-27">
      <a class="sample-nav prev" href="#t2i-sample-26" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/765.jpg' | relative_url }}" alt="Text-to-image sample 27" />
      <a class="sample-nav next" href="#t2i-sample-28" aria-label="Next sample">›</a>
      <figcaption>Rows of grilled skewers are packed tightly on a metal tray, coated in glossy red seasoning, herbs, and chili flakes. The close-up view highlights the rich sauce, charred edges, and spicy street-food appeal of the dish.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-28">
      <a class="sample-nav prev" href="#t2i-sample-27" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/807.jpg' | relative_url }}" alt="Text-to-image sample 28" />
      <a class="sample-nav next" href="#t2i-sample-29" aria-label="Next sample">›</a>
      <figcaption>Thick steam rises from a large pot filled with stew, meat, vegetables, and rich brown broth. The close view captures active cooking, with bubbling liquid and softened ingredients creating a hot, hearty, and aromatic kitchen scene.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-29">
      <a class="sample-nav prev" href="#t2i-sample-28" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/808.jpg' | relative_url }}" alt="Text-to-image sample 29" />
      <a class="sample-nav next" href="#t2i-sample-30" aria-label="Next sample">›</a>
      <figcaption>Small white flowers bloom in dense clusters among glossy green leaves, with many speckled petals and unopened buds visible. The close-up composition emphasizes delicate natural patterns, soft lighting, and the fresh detail of the flowering plant.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-30">
      <a class="sample-nav prev" href="#t2i-sample-29" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/850.jpg' | relative_url }}" alt="Text-to-image sample 30" />
      <a class="sample-nav next" href="#t2i-sample-31" aria-label="Next sample">›</a>
      <figcaption>A brilliant sun shines above a winding mountain road bordered by lush green plants. Distant ridges fade toward the blue horizon, while the intense light and clear sky create a bright, open, high-altitude travel scene.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-31">
      <a class="sample-nav prev" href="#t2i-sample-30" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/884.jpg' | relative_url }}" alt="Text-to-image sample 31" />
      <a class="sample-nav next" href="#t2i-sample-32" aria-label="Next sample">›</a>
      <figcaption>A vivid red boat rests on pale sand in front of calm turquoise water and a clear horizon. The simple composition emphasizes strong color contrast, seaside quietness, and the unusual stillness of a boat pulled ashore.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-32">
      <a class="sample-nav prev" href="#t2i-sample-31" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/h2.jpg' | relative_url }}" alt="Text-to-image sample 32" />
      <a class="sample-nav next" href="#t2i-sample-33" aria-label="Next sample">›</a>
      <figcaption>A man in a gray polo shirt stands on a balcony or rooftop, looking thoughtfully to the side. A blurred city skyline behind him gives the portrait an urban setting and a calm, reflective mood.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-33">
      <a class="sample-nav prev" href="#t2i-sample-32" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/h4.jpg' | relative_url }}" alt="Text-to-image sample 33" />
      <a class="sample-nav next" href="#t2i-sample-34" aria-label="Next sample">›</a>
      <figcaption>A close selfie shows a man wearing a dark bucket hat and light blue shirt while seated inside a vehicle. Black seats, side windows, and an orange curtain provide context for the casual travel setting.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-34">
      <a class="sample-nav prev" href="#t2i-sample-33" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/h5.jpg' | relative_url }}" alt="Text-to-image sample 34" />
      <a class="sample-nav next" href="#t2i-sample-35" aria-label="Next sample">›</a>
      <figcaption>An anime-style girl with long dark hair wears a purple patterned outfit with lace details, posed against a dreamy sky. Large moonlike circles, stars, and soft gradients create a romantic fantasy illustration style.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-35">
      <a class="sample-nav prev" href="#t2i-sample-34" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/h6.jpg' | relative_url }}" alt="Text-to-image sample 35" />
      <a class="sample-nav next" href="#t2i-sample-36" aria-label="Next sample">›</a>
      <figcaption>A woman with curly hair smiles gently in a very dark outdoor setting, lit softly from the front. The nearly black background isolates her face and shoulders, creating an intimate portrait with understated nighttime atmosphere.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-36">
      <a class="sample-nav prev" href="#t2i-sample-35" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/h7.jpg' | relative_url }}" alt="Text-to-image sample 36" />
      <a class="sample-nav next" href="#t2i-sample-37" aria-label="Next sample">›</a>
      <figcaption>Two men in gray shirts stand shoulder to shoulder in front of a green screen. One smiles broadly while the other gives a restrained expression, creating a casual studio portrait with contrasting personalities.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-37">
      <a class="sample-nav prev" href="#t2i-sample-36" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/h8.jpg' | relative_url }}" alt="Text-to-image sample 37" />
      <a class="sample-nav next" href="#t2i-sample-38" aria-label="Next sample">›</a>
      <figcaption>Two adults sit at a restaurant table behind a birthday cake topped with lit candles. Warm indoor lighting, surrounding dishes, cups, and desserts suggest a celebratory meal shared in a relaxed dining setting.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-38">
      <a class="sample-nav prev" href="#t2i-sample-37" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/h9.jpg' | relative_url }}" alt="Text-to-image sample 38" />
      <a class="sample-nav next" href="#t2i-sample-39" aria-label="Next sample">›</a>
      <figcaption>A fluffy white cat sits indoors with wide dark eyes, pink ears, and crossed front paws. The softly blurred background and centered framing make the cat's round face and clean white fur especially prominent.</figcaption>
    </figure>
    <figure class="sample-slide" id="t2i-sample-39">
      <a class="sample-nav prev" href="#t2i-sample-38" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/t2i_samples/h10.jpg' | relative_url }}" alt="Text-to-image sample 39" />
      <a class="sample-nav next" href="#t2i-sample-1" aria-label="Next sample">›</a>
      <figcaption>Two young women in coordinated school-style outfits with red bows take a close selfie outdoors. Trees, bicycles, and paved walkways in the background suggest a campus or park setting on a bright day.</figcaption>
    </figure>
  </div>
  <div class="sample-dots" aria-label="Text-to-image sample shortcuts">
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

### 4.2 Image-and-text-to-text {#image-text-to-text}

After training on about **5M** image-text pairs, the model shows preliminary image-captioning ability and can generate text conditioned on image content.

<div class="sample-carousel" aria-label="Image-conditioned text generation sample gallery">
  <div class="sample-track">
    <figure class="sample-slide" id="it2t-sample-1">
      <a class="sample-nav prev" href="#it2t-sample-36" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/8.png' | relative_url }}" alt="Image-conditioned text generation sample 1" />
      <a class="sample-nav next" href="#it2t-sample-2" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a woman wearing an oversized long down jacket with a plain collar. The jacket features a letter ``A'' design on it.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-2">
      <a class="sample-nav prev" href="#it2t-sample-1" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/11.png' | relative_url }}" alt="Image-conditioned text generation sample 2" />
      <a class="sample-nav next" href="#it2t-sample-3" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a person wearing a black jacket and sunglasses, standing on a street.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-3">
      <a class="sample-nav prev" href="#it2t-sample-2" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/54.png' | relative_url }}" alt="Image-conditioned text generation sample 3" />
      <a class="sample-nav next" href="#it2t-sample-4" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a meeting room with several people, around six to eight, sitting around a long conference table and holding a meeting. A screen is mounted on the wall, and bowls, cups, and pens are placed on the table.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-4">
      <a class="sample-nav prev" href="#it2t-sample-3" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/57.png' | relative_url }}" alt="Image-conditioned text generation sample 4" />
      <a class="sample-nav next" href="#it2t-sample-5" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a person wearing a grey hoodie with a purple design on the front. The background includes a white wall with a blue design.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-5">
      <a class="sample-nav prev" href="#it2t-sample-4" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/58.png' | relative_url }}" alt="Image-conditioned text generation sample 5" />
      <a class="sample-nav next" href="#it2t-sample-6" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows several people playing a ball game on a court. The background includes a blue wall with some text, and the ground is also blue.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-6">
      <a class="sample-nav prev" href="#it2t-sample-5" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/59.png' | relative_url }}" alt="Image-conditioned text generation sample 6" />
      <a class="sample-nav next" href="#it2t-sample-7" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a person wearing a black suit with a white shirt underneath, seated.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-7">
      <a class="sample-nav prev" href="#it2t-sample-6" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/60.png' | relative_url }}" alt="Image-conditioned text generation sample 7" />
      <a class="sample-nav next" href="#it2t-sample-8" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a cluster of ancient buildings with old-style architecture, featuring several old structures packed closely together.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-8">
      <a class="sample-nav prev" href="#it2t-sample-7" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/61.png' | relative_url }}" alt="Image-conditioned text generation sample 8" />
      <a class="sample-nav next" href="#it2t-sample-9" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image is a cityscape shot at dusk, showing an urban night scene with many tall high-rise buildings.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-9">
      <a class="sample-nav prev" href="#it2t-sample-8" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/84.png' | relative_url }}" alt="Image-conditioned text generation sample 9" />
      <a class="sample-nav next" href="#it2t-sample-10" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a large football (soccer) stadium with the field inside, resembling an outdoor stadium. Another row of stands is visible on the other side of the field.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-10">
      <a class="sample-nav prev" href="#it2t-sample-9" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/89.png' | relative_url }}" alt="Image-conditioned text generation sample 10" />
      <a class="sample-nav next" href="#it2t-sample-11" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe the content of this image in detail.<br /><strong>Output:</strong> The image shows a person using a pen to write on a piece of paper.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-11">
      <a class="sample-nav prev" href="#it2t-sample-10" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/12.png' | relative_url }}" alt="Image-conditioned text generation sample 11" />
      <a class="sample-nav next" href="#it2t-sample-12" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> What does this image depict? Give a comprehensive description.<br /><strong>Output:</strong> The image shows a woman with long, dark hair wearing a black top. The background is a neutral color, with a window and a pink wall visible behind her. She appears to be standing indoors.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-12">
      <a class="sample-nav prev" href="#it2t-sample-11" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/13.png' | relative_url }}" alt="Image-conditioned text generation sample 12" />
      <a class="sample-nav next" href="#it2t-sample-13" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> What does this image depict? Give a comprehensive description.<br /><strong>Output:</strong> The image shows a portrait of a young woman with blonde hair, wearing a white top and seated at a student desk with an open laptop. She appears to be speaking while using the laptop.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-13">
      <a class="sample-nav prev" href="#it2t-sample-12" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/15.png' | relative_url }}" alt="Image-conditioned text generation sample 13" />
      <a class="sample-nav next" href="#it2t-sample-14" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> What does this image depict? Give a comprehensive description.<br /><strong>Output:</strong> The image shows a gravestone with English text carved on it.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-14">
      <a class="sample-nav prev" href="#it2t-sample-13" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/36.png' | relative_url }}" alt="Image-conditioned text generation sample 14" />
      <a class="sample-nav next" href="#it2t-sample-15" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image shows a cluster of modern city buildings, including tall skyscrapers with distinctive and futuristic architectural designs.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-15">
      <a class="sample-nav prev" href="#it2t-sample-14" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/43.png' | relative_url }}" alt="Image-conditioned text generation sample 15" />
      <a class="sample-nav next" href="#it2t-sample-16" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image is a medium shot of a young East Asian woman with long, straight dark hair, wearing a pink top and a necklace. She is posing with her hands raised near her face, making a gesture with her fingers.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-16">
      <a class="sample-nav prev" href="#it2t-sample-15" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/44.png' | relative_url }}" alt="Image-conditioned text generation sample 16" />
      <a class="sample-nav next" href="#it2t-sample-17" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image shows a blue card or cover with a printed image and some text on it.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-17">
      <a class="sample-nav prev" href="#it2t-sample-16" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/46.png' | relative_url }}" alt="Image-conditioned text generation sample 17" />
      <a class="sample-nav next" href="#it2t-sample-18" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image shows three people standing in an open outdoor area. They are wearing colored clothing, and one of them appears to be carrying something on their back.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-18">
      <a class="sample-nav prev" href="#it2t-sample-17" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/67.png' | relative_url }}" alt="Image-conditioned text generation sample 18" />
      <a class="sample-nav next" href="#it2t-sample-19" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image shows a mountain range with a massive peak. The surface of the mountain reveals distinctive textures and patterns.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-19">
      <a class="sample-nav prev" href="#it2t-sample-18" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/68.png' | relative_url }}" alt="Image-conditioned text generation sample 19" />
      <a class="sample-nav next" href="#it2t-sample-20" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image shows a man wearing a plaid shirt, working in a studio or workshop. He is focused on an object on a white surface in front of him.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-20">
      <a class="sample-nav prev" href="#it2t-sample-19" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/69.png' | relative_url }}" alt="Image-conditioned text generation sample 20" />
      <a class="sample-nav next" href="#it2t-sample-21" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image shows a newlywed couple at a wedding, dressed in wedding attire.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-21">
      <a class="sample-nav prev" href="#it2t-sample-20" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/70.png' | relative_url }}" alt="Image-conditioned text generation sample 21" />
      <a class="sample-nav next" href="#it2t-sample-22" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> What is shown in this image? Please explain thoroughly.<br /><strong>Output:</strong> The image is a cartoon-style drawing of a character dressed in a colored outfit, with a body showing green and yellow stripes. The character is touching its face with one hand.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-22">
      <a class="sample-nav prev" href="#it2t-sample-21" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/26.png' | relative_url }}" alt="Image-conditioned text generation sample 22" />
      <a class="sample-nav next" href="#it2t-sample-23" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows musicians performing with instruments at a ceremony. The background includes a floor and a wall.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-23">
      <a class="sample-nav prev" href="#it2t-sample-22" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/74.png' | relative_url }}" alt="Image-conditioned text generation sample 23" />
      <a class="sample-nav next" href="#it2t-sample-24" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows a black sleeveless T-shirt with a printed design on the front.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-24">
      <a class="sample-nav prev" href="#it2t-sample-23" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/75.png' | relative_url }}" alt="Image-conditioned text generation sample 24" />
      <a class="sample-nav next" href="#it2t-sample-25" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows a person wearing a white blouse over a black and white striped skirt.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-25">
      <a class="sample-nav prev" href="#it2t-sample-24" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/78.png' | relative_url }}" alt="Image-conditioned text generation sample 25" />
      <a class="sample-nav next" href="#it2t-sample-26" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows a white Toyota car with a sleek design, featuring prominent front and side headlights.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-26">
      <a class="sample-nav prev" href="#it2t-sample-25" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/79.png' | relative_url }}" alt="Image-conditioned text generation sample 26" />
      <a class="sample-nav next" href="#it2t-sample-27" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows a man wearing a black baseball cap and a grey shirt. The focus is on the cap, and the background is a plain, neutral grey.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-27">
      <a class="sample-nav prev" href="#it2t-sample-26" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/80.png' | relative_url }}" alt="Image-conditioned text generation sample 27" />
      <a class="sample-nav next" href="#it2t-sample-28" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows a necklace with a red, ball-shaped pendant.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-28">
      <a class="sample-nav prev" href="#it2t-sample-27" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/81.png' | relative_url }}" alt="Image-conditioned text generation sample 28" />
      <a class="sample-nav next" href="#it2t-sample-29" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows a person standing on a paved area with a scenic green backdrop.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-29">
      <a class="sample-nav prev" href="#it2t-sample-28" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/90.png' | relative_url }}" alt="Image-conditioned text generation sample 29" />
      <a class="sample-nav next" href="#it2t-sample-30" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows an old building with distinctive traditional Chinese-style architecture. It features traditional entrance details, and an iron structure with a roof is visible above the building.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-30">
      <a class="sample-nav prev" href="#it2t-sample-29" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/91.png' | relative_url }}" alt="Image-conditioned text generation sample 30" />
      <a class="sample-nav next" href="#it2t-sample-31" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows a decorative ornament in the shape of a flower.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-31">
      <a class="sample-nav prev" href="#it2t-sample-30" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/94.png' | relative_url }}" alt="Image-conditioned text generation sample 31" />
      <a class="sample-nav next" href="#it2t-sample-32" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Describe what you see in this picture.<br /><strong>Output:</strong> The image shows several people playing volleyball, with one of them jumping up. The scene takes place under a clear afternoon sky.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-32">
      <a class="sample-nav prev" href="#it2t-sample-31" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/48.png' | relative_url }}" alt="Image-conditioned text generation sample 32" />
      <a class="sample-nav next" href="#it2t-sample-33" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Generate a detailed caption for this image.<br /><strong>Output:</strong> The image shows a soccer player in action during a match, wearing a numbered uniform. The player appears to be running and kicking the ball on the field.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-33">
      <a class="sample-nav prev" href="#it2t-sample-32" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/50.png' | relative_url }}" alt="Image-conditioned text generation sample 33" />
      <a class="sample-nav next" href="#it2t-sample-34" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Generate a detailed caption for this image.<br /><strong>Output:</strong> The image shows a black T-shirt with a design featuring a blue cat printed on the front.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-34">
      <a class="sample-nav prev" href="#it2t-sample-33" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/51.png' | relative_url }}" alt="Image-conditioned text generation sample 34" />
      <a class="sample-nav next" href="#it2t-sample-35" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Generate a detailed caption for this image.<br /><strong>Output:</strong> The image shows a young boy wearing a grey vest over a white shirt, with a blue accent.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-35">
      <a class="sample-nav prev" href="#it2t-sample-34" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/96.png' | relative_url }}" alt="Image-conditioned text generation sample 35" />
      <a class="sample-nav next" href="#it2t-sample-36" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Generate a detailed caption for this image.<br /><strong>Output:</strong> The image shows a young girl with straight hair, wearing a yellow crop top and white shorts. She is making a fist with one hand and jumping, set against a blue background.</figcaption>
    </figure>
    <figure class="sample-slide" id="it2t-sample-36">
      <a class="sample-nav prev" href="#it2t-sample-35" aria-label="Previous sample">‹</a>
      <img src="{{ '/assets/it2t_samples/97.png' | relative_url }}" alt="Image-conditioned text generation sample 36" />
      <a class="sample-nav next" href="#it2t-sample-1" aria-label="Next sample">›</a>
      <figcaption><strong>Prompt:</strong> Generate a detailed caption for this image.<br /><strong>Output:</strong> The image shows the stands of a large soccer stadium. The field and running track are visible, with green grass surrounding the stands and a few spectators present.</figcaption>
    </figure>
  </div>
  <div class="sample-dots" aria-label="Image-conditioned text generation sample shortcuts">
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

> *Note on images.* The images used in the image-and-text-to-text examples are generated by **external image-generation models from ground-truth captions**, to avoid copyright issues associated with real photographs.

### 4.3 Text-to-text {#text-to-text}

Although trained with only about **1B** text tokens, the unified model retains reasonably coherent text-continuation ability across conversational, narrative, expository, technical, and article-style prompts.

---

## 5. Future Experiments: From Feasibility to Quantitative Comparison {#future-experiments}

The current results primarily validate the **feasibility** of the proposed unified modeling scheme: a single model can acquire text generation, image generation, and preliminary image-text understanding ability under the same pretraining recipe. The next question is whether this unified model, after scaling pretraining and adding necessary SFT/RL post-training, can outperform standard alternatives. To answer this, we plan two controlled quantitative studies.

**(Q1) Under controlled settings, does unified multimodal pretraining improve generation (pixel output)?**
We will first compare against a **standard MMDiT** baseline. This baseline follows the conventional text-to-image setup, where text is used only as a conditioning input and is not part of the joint latent-variable modeling. The comparison will match **data, parameter count, and number of training samples**, isolating the effect of the training paradigm itself and testing whether modeling a joint latent distribution with a shared MMDiT yields measurable gains in generation quality.

**(Q2) Does unified multimodal pretraining also improve understanding (text output)?**
Beyond generation quality, we will evaluate whether shared-MMDiT modeling improves multimodal understanding and compare it systematically with other unified modeling paradigms.

Future work will also report scaling behavior, controlled matched comparisons, and results on standard generation and understanding benchmarks.

---

## 6. Roadmap: More Modalities under a Shared Joint Prior {#roadmap}

The framework naturally extends to more modalities by introducing the corresponding block layouts and attention rules. Potential extensions include:

- **Video.** Temporally blocked latents fit the block-causal structure naturally: each frame or clip becomes a block, with causal structure across blocks and bidirectional modeling within each block. This form supports interleaved image, text, and video generation and understanding in a single sequence.
- **Audio.** Continuous audio latents can be introduced as another segment type, enabling text-audio and video-audio alignment for tasks such as captioning, narration, and sound-conditioned generation.
- **Action.** Continuous action latents, such as trajectories or control signals, can be incorporated into the shared block-causal MMDiT, supporting joint reasoning over perception and action and providing an interface for world-model-style learning.
<!-- - **Intermediate reasoning latents.** Since text and other modalities are organized as causal blocks, the model could in principle generate intermediate reasoning latents before producing images, videos, or actions, providing an explicit reasoning path for multimodal generation. -->

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

Cola DLM project page: <https://hongcanguo.github.io/Cola-DLM/>
