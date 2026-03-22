# Improving Clinical Factuality in Medical Visual Question Answering via Multi-Candidate Answer Generation and Reranking
## Overview

This repository implements a unified **generation–selection framework** for open-ended Medical Visual Question Answering (Med-VQA).

Unlike conventional single-generation approaches, this work formulates Med-VQA as a **multi-candidate answer generation and reranking problem**, where:

- Multiple candidate answers are first generated for each image-question pair
- A selection mechanism is then applied to choose the best answer

We explore several answer selection strategies, including:

- Multi-Candidate First
- Prompt-based LLM Selector
- Reference-Distilled Pairwise Reranker (proposed)

The proposed reranker explicitly models the **relative quality differences among candidate answers**, improving:

- factual consistency
- spatial reasoning
- coverage of key findings

---

## 🔬 Key Contributions

- Reformulate open-ended Med-VQA as a **generate–select problem**
- Propose a **reference-distilled pairwise reranker**
- Provide a unified comparison of multiple candidate selection strategies
- Conduct extensive experiments on the PubMedVision dataset
- Demonstrate consistent improvements over single-generation baselines

  ## Project Structure
.
├── src/                          # Core source code
├── scripts/                      # Training / evaluation scripts
├── LLaMA-Factory/               # Base model training framework (LoRA fine-tuning)
├── examples/                    # Example usage
├── assets/                      # Figures / visualization
├── docker/                      # Docker configs
├── tools/                       # Utility scripts
├── tests/                       # Testing scripts
├── tests_v1/                    # Additional experiments
├── llamaboard_config/           # Configuration for visualization
├── *.py                         # Main pipeline scripts
## Installation



## Dataset

Experiments are conducted on the **PubMedVision** dataset.

Due to licensing restrictions, the dataset is not included in this repository.

Please follow the official dataset instructions for access.
