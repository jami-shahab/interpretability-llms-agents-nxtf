# XAI Refresher Overview

## Introduction

Welcome to the **XAI Refresher** implementation of the Interpretability for LLMs and Agents Bootcamp.
This folder covers foundational and advanced techniques in Explainable AI (XAI), with a focus on
post-hoc explanation methods for both traditional neural networks and modern vision-language models
(VLMs). We explore how to make model decisions interpretable through feature attribution,
segmentation-based perturbations, concept decomposition, and gradient-based visualization.

## Prerequisites

Before diving into the materials, ensure you have the following:

- Python 3.10 or higher
- PyTorch 2.x
- Basic familiarity with neural networks and image classification
- Familiarity with Python and Jupyter notebooks
- A CUDA-capable GPU is recommended for the concept grounding notebook and the perturbation notebooks
- Additional libraries for notebooks 5, 6, 7 & 8: `captum`, `transformers`, `datasets`, `bertviz` (installed via the same `uv` dependency group)

## Notebooks

The following Jupyter notebooks are provided in this folder:

1. **[LIME](lime.ipynb)** — Covers the LIME (Local Interpretable Model-agnostic Explanations)
   framework for image, tabular, and text models. Includes LORE (rule-based local explanations
   for tabular data) and DSEG-LIME (SAM-powered data-driven segmentation for richer image
   explanations).

2. **[SHAP](shap.ipynb)** — Introduces SHAP (SHapley Additive exPlanations) with KernelExplainer
   applied to a PyTorch MLP trained on the UCI Credit Card Default dataset. Covers SHAP value
   computation, summary plots, and how to interpret additive feature contributions.

3. **[CLIP Interpretability](clip.ipynb)** — Explores concept-based interpretability for
   vision-language models using CLIP. Covers representation-level analysis, Grad-CAM and
   EigenCAM heatmaps, and how embedding-space geometry relates to model decisions.

4. **[Concept Grounding](concept_grounding.ipynb)** — Demonstrates how to extract and decompose
   hidden-state features from LLaVA (7B) using Symmetric Non-negative Matrix Factorization
   (SNMF). Covers concept dictionary learning, multimodal grounding (text + image), and
   local interpretations per sample on COCO.

5. **[Perturbation & Robustness — Vision](perturbation_robustness_captum_image.ipynb)** —
   Covers perturbation-based attribution for image classifiers using the Captum library.
   Implements Occlusion, Feature Ablation, and Noise Tunnel (SmoothGrad) on a ResNet-18
   model. Evaluates explanation quality with the Infidelity and Sensitivity metrics.
   *Do this notebook before the text version.*

6. **[Perturbation & Robustness — Text + Bias](perturbation_robustness_and_bias_text.ipynb)** —
   Mirrors notebook 5 for transformer-based text classifiers (BERT fine-tuned on SST-2).
   Implements token ablation and gradient attribution, then extends to explanation robustness
   under paraphrase, Counterfactual Fairness Distance (CFD) for bias probing, and a Masked
   Language Model pronoun prediction probe to detect occupational gender stereotypes.
   *Do after the vision perturbation notebook.*

7. **[TCAV — Concept-Level Interpretability](tcav_concept_sensitivity.ipynb)** —
   Implements Testing with Concept Activation Vectors (TCAV) for a BERT sentiment classifier.
   Covers CAV training via logistic regression on hidden-layer activations, directional
   derivative computation, and TCAV score analysis across all 13 transformer layers.
   Uses real SST-2 sentences (loaded from HuggingFace) as concept examples for stable CAVs.
   Includes a profession concept probe that reveals BERT-SST2's spurious association between
   professional-activity sentence structure and positive sentiment — independently corroborating
   the bias findings from notebook 6 via a completely different method.
   *Do after notebook 6.*

8. **[Attention vs Attribution](attention_vs_attribution.ipynb)** — Investigates whether
   attention weights are a reliable proxy for feature importance. Visualizes self-attention
   patterns across all heads and layers using BertViz, then computes gradient-based token
   attribution (embedding gradient L2 norm) on the same BERT-SST2 model. Quantifies
   attention–attribution alignment via Pearson correlation across all 12 layers, showing
   that early layers are strongly misaligned while mid-depth layers exhibit slight positive
   alignment. Includes a clause-level counterfactual experiment demonstrating that "but" vs
   "and" connectives reshape both the prediction and the attribution signal.
   *Do after notebook 7.*

### Notebooks 5, 6, 7 & 8: Cross-Notebook Connection

Notebooks 5–8 build on each other progressively — from perturbation methods to concept-level probing to the fundamental question of whether attention can substitute for attribution:

| Concept                    | Vision (5)                        | Text + Bias (6)                       | TCAV (7)                          | Attn vs Attr (8)               |
| -------------------------- | --------------------------------- | ------------------------------------- | --------------------------------- | ------------------------------ |
| Perturbation attribution   | Occlusion (patch masking)         | Token ablation (`[MASK]`)             | —                                 | —                              |
| Gradient attribution       | Saliency (input gradients)        | Embedding gradient (L2 norm)          | Directional derivative            | Embedding gradient (L2 norm)   |
| Robustness / faithfulness  | Infidelity + Sensitivity (Captum) | Explanation distance (L2)             | —                                 | Layer-wise correlation         |
| Concept-level probing      | —                                 | —                                     | CAV + TCAV scores (13 layers)     | —                              |
| Attention analysis         | —                                 | —                                     | —                                 | BertViz (model_view, head_view)|
| Bias / fairness            | —                                 | CFD + MLM pronoun probe               | Profession concept (TCAV=1.0)     | Counterfactual clauses         |

**Key shared insight:** perturbation-based methods are more faithful (causal) but slower;
gradient methods are faster but measure magnitude only; attention weights are the fastest but
are not reliably aligned with prediction sensitivity — especially in early layers.

## Package Dependencies

The Concept Grounding notebook requires Java to be installed. On Linux, you can install it using:

```bash
sudo apt update
sudo apt install -y default-jre
```

## Resources

For further reading on the methods covered in this module:

**LIME & Variants**
- Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "'Why Should I Trust You?': Explaining the Predictions of Any Classifier." *KDD*, 2016.
- Guidotti, Riccardo, et al. "Local rule-based explanations of black box decision systems." *arXiv:1805.10820*, 2018. *(LORE)*
- Knab, Patrick, Sascha Marton, and Christian Bartelt. "Beyond Pixels: Enhancing LIME with Hierarchical Features and Segmentation Foundation Models." *arXiv:2403.07733*, 2024. *(DSEG-LIME)*

**SHAP**
- Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions." *NeurIPS*, 30, 2017.

**CLIP & Gradient-based Vision**
- Selvaraju, Ramprasaath R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV*, 2017.
- Muhammad, Mohammed Bany, and Mohammed Yeasin. "Eigen-CAM: Class Activation Map using Principal Components." *IJCNN*, 2020.

**Perturbation & Robustness**
- Zeiler, Matthew D., and Rob Fergus. "Visualizing and Understanding Convolutional Networks." *ECCV*, 2014. *(Occlusion-based attribution)*
- Smilkov, Daniel, et al. "SmoothGrad: removing noise by adding noise." *arXiv:1706.03825*, 2017.
- Yeh, Chih-Kuan, et al. "On the (In)fidelity and Sensitivity of Explanations." *NeurIPS*, 32, 2019.
- Samek, Wojciech, et al. "Evaluating the visualization of what a deep neural network has learned." *IEEE TNNLS*, 28(11), 2016.
- Atmakuri, Shriya, et al. "Robustness of Explanation Methods for NLP Models." *arXiv:2206.12284*, 2022.

**Bias Probing**
- Kurita, Keita, Nidhi Vyas, Ayush Pareek, Alan W Black, and Yulia Tsvetkov. "Measuring Bias in Contextualized Word Representations." *ACL Workshop on Gender Bias in NLP*, 2019.

**TCAV — Concept-Level Interpretability**
- Kim, Been, et al. "Interpretability beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)." *ICML*, 2018.
- Alain, Guillaume, and Yoshua Bengio. "Understanding intermediate layers using linear classifier probes." *arXiv:1610.01644*, 2016. *(CAV foundation)*
- Tenney, Ian, Dipanjan Das, and Ellie Pavlick. "BERT Rediscovers the Classical NLP Pipeline." *ACL*, 2019.

**Attention vs Attribution**
- Jain, Sarthak, and Byron C. Wallace. "Attention is not Explanation." *NAACL*, 2019.
- Wiegreffe, Sarah, and Yuval Pinter. "Attention is not not Explanation." *EMNLP-IJCNLP*, 2019.
- Vig, Jesse. "A Multiscale Visualization of Attention in the Transformer Model." *ACL: System Demonstrations*, 2019. *(BertViz)*
- Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. "Axiomatic Attribution for Deep Networks." *ICML*, 2017. *(Integrated Gradients)*
- Vaswani, Ashish, et al. "Attention is All You Need." *NeurIPS*, 30, 2017.

**Concept Grounding in VLMs**
- Parekh, Jayneel, et al. "A Concept-based Explainability Framework for Large Multimodal Models." *NeurIPS*, 37, 2024.

## Getting Started

1. From the **root of the repository**, create a virtual environment and install the
   `xai-refresher` dependency group using `uv`:

   ```bash
   uv sync --group xai-refresher
   ```

   This creates a `.venv` in the repo root and installs all packages needed for this module
   (PyTorch, SHAP, LIME, Grad-CAM, SAM, etc.).

2. Activate the environment:

   ```bash
   source .venv/bin/activate
   ```

3. Start with **[lime.ipynb](lime.ipynb)** for a ground-up introduction to post-hoc explanation
   with LIME and its variants.

4. Proceed to **[shap.ipynb](shap.ipynb)** to explore Shapley-value-based attribution on a
   tabular classification task.

5. Move to **[clip.ipynb](clip.ipynb)** to see how gradient-based and representation-level
   explanations apply to vision-language models.

6. Continue with **[perturbation_robustness_captum_image.ipynb](perturbation_robustness_captum_image.ipynb)**
   to explore perturbation-based attribution and explanation faithfulness metrics for vision
   models using Captum.

7. Follow up with **[perturbation_robustness_and_bias_text.ipynb](perturbation_robustness_and_bias_text.ipynb)**
   to apply the same perturbation framework to NLP, and extend it to robustness evaluation
   and bias probing. Note: this notebook downloads `bert-base-uncased` on first run.

8. Continue with **[tcav_concept_sensitivity.ipynb](tcav_concept_sensitivity.ipynb)** to
   probe BERT-SST2 with concept activation vectors across all transformer layers, and
   uncover its spurious profession-sentiment association.

9. Continue with **[attention_vs_attribution.ipynb](attention_vs_attribution.ipynb)** to
   directly compare attention weights against gradient-based attribution, and investigate
   whether attention can serve as a reliable explanation.

10. Finish with **[concept_grounding.ipynb](concept_grounding.ipynb)** for a deep dive into
    concept decomposition and grounding in LLaVA. Note: this notebook requires a GPU and
    will download model weights on first run.
