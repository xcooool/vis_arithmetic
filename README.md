# Multi-modal Reasoning with LLMs for Visual Semantic Arithmetic

## Overview
We define a formulation to represent the meta-data and design two semantic arithmetic tasks:
- **Two-term subtraction task**: denote the task as **object − subject = relation**, which requires the models to infer the relation with subject–object pair input (implemented as a multiple-choice question).
- **Three-term operation task**: denote the task as **object1 − subject1 + subject2 = object2**, which requires the models to generate the text response representing object2 with three-term input (analogy-style).

Inspired by ConceptNet, we construct the **Image-Relation-Pair Dataset (IRPD)** for systematic evaluation. Moreover, in contrast to embedding arithmetic baselines, we propose **Semantic Arithmetic Reinforcement Fine-Tuning (SAri-RFT)**, which post-trains an LVLM by incorporating reinforcement learning with verifiable rewards with a newly designed verifiable reward function via **Group Relative Policy Optimization (GRPO)**.

## IRPD Dataset
- **IRPD (Google Drive)**: [download link](https://drive.google.com/drive/folders/1LJr9u1LBgSUnblfroRQ2sDd-6jPJoEqm?usp=sharing)
- **Dataset generation pipeline**: see `IRPD_dataset/`

## Code Structure
We provide four main directories:
- **`embedding_arithmetic/`**: baselines for embedding arithmetic (e.g., ZeroCap, ImageBind, LanguageBind).
- **`evalution/`**: evaluation code for IRPD and Visual7W-Telling.
- **`IRPD_dataset/`**: IRPD dataset generation pipeline.
- **`sari_rft/`**: SAri-RFT training methods, including GRPO for two tasks and SFT.

## Figures
### IRPD generation pipeline
![IRPD generation pipeline](figures/pipeline.png)

### Qualitative / results visualization
![Qualitative / results visualization](figures/quality_res.png)

### Flux dataset illustration
![Flux dataset illustration](figures/flux_dataset.png)

## Quick Start
Please refer to the subdirectories above for task-specific scripts and instructions. Typical workflow:
- **Dataset**: download IRPD from the Google Drive link above (or build it via `IRPD_dataset/`), then set dataset paths in the corresponding scripts.
- **Training**: run SAri-RFT under `sari_rft/` (GRPO/SFT).
- **Evaluation**: run evaluation under `evalution/`.

## Acknowledgement
We sincerely thank [Visual-RFT](https://github.com/Liuziyu77/Visual-RFT) and [ZeroCap](https://github.com/YoadTew/zero-shot-image-to-text) for their open-source resources.
