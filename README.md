# cogintro

<div align="center">

![GitHub top language](https://img.shields.io/github/languages/top/CogitoNTNU/cogintro)
![GitHub language count](https://img.shields.io/github/languages/count/CogitoNTNU/cogintro)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Cogito NTNU's Introduction Group**

</div>

## Overview

This repository contains materials and projects from Cogito NTNU's introduction group. Unlike regular Cogito projects that focus on a single target throughout the semester, CogIntro covers a broad range of AI/ML topics to maximize learning. The group of 8 members met twice a week in afternoon sessions, with each subproject spanning 1-4 weeks. The progression went from foundational tooling through increasingly complex applications, ending with a deep learning project on HPC infrastructure.

### Course Progression

```mermaid
flowchart LR
    A[Project Start] --> B[Git]
    B --> C[ML Fundamentals]
    C --> D[Flappy Bird RL]
    D --> E[LLM Chatbot]
    E --> F[Tumor Segmentation]
    F --> G[Project Presentation]

    style A fill:#374151,stroke:#6b7280
    style G fill:#374151,stroke:#6b7280
```

## Projects

### Fundamentals

Terminal usage, Git workflows, and introductory ML concepts via Kaggle notebooks.

### Flappy Bird Agent

A reinforcement learning agent trained to play Flappy Bird using the OpenAI Gym framework.

<div align="center">
<a href="src/flappy-bird-gym">
<img src="docs/images/flappy-bird-agent.gif" width="200" alt="Flappy Bird RL Agent">
</a>

[Project folder →](src/flappy-bird-gym)
</div>

### LLM Chatbot

A conversational chatbot built with OpenAI's Responses API.

<div align="center">
<a href="src/large-language-models">
<img src="docs/images/chatbot-companion.jpeg" width="400" alt="LLM Chatbot">
</a>

[Project folder →](src/large-language-models)
</div>

### Tumor Segmentation on IDUN

Medical image segmentation using U-Net architecture, trained on NTNU's IDUN HPC cluster via SLURM. The model segments tumor regions from PET/CT scans using k-fold cross-validation.

<div align="center">
<a href="src/tumor-segmentation">
<img src="docs/images/tumor-segmentation-results.png" width="600" alt="Tumor Segmentation Results from W&B">
</a>

[Project folder →](src/tumor-segmentation)
</div>

## Prerequisites

- **Git**: [Download Git](https://git-scm.com/downloads)
- **Python 3.12**: [Download Python](https://www.python.org/downloads/)
- **UV**: Python package manager. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)

## Getting Started

```sh
git clone https://github.com/CogitoNTNU/cogintro.git
cd cogintro
uv sync
```

For development:
```sh
uv run pre-commit install
```

## Repository Structure

```
cogintro/
├── src/
│   ├── flappy-bird-gym/     # RL environment and agent
│   ├── large-language-models/  # OpenAI chatbot notebook
│   └── tumor-segmentation/  # U-Net model and SLURM scripts
├── docs/
└── tests/
```

## Team

<table align="center">
    <tr>
        <td align="center">
            <a href="https://github.com/maiahi">
              <img src="https://github.com/maiahi.png?size=100" width="100px;" alt="maiahi"/><br />
              <sub><b>maiahi</b></sub>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/AlMinaDO">
              <img src="https://github.com/AlMinaDO.png?size=100" width="100px;" alt="AlMinaDO"/><br />
              <sub><b>AlMinaDO</b></sub>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/arlindakm">
              <img src="https://github.com/arlindakm.png?size=100" width="100px;" alt="arlindakm"/><br />
              <sub><b>arlindakm</b></sub>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/flatval">
              <img src="https://github.com/flatval.png?size=100" width="100px;" alt="flatval"/><br />
              <sub><b>flatval</b></sub>
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/Knolldus">
              <img src="https://github.com/Knolldus.png?size=100" width="100px;" alt="Knolldus"/><br />
              <sub><b>Knolldus</b></sub>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/Jarandvs">
              <img src="https://github.com/Jarandvs.png?size=100" width="100px;" alt="Jarandvs"/><br />
              <sub><b>Jarandvs</b></sub>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/ApatShe">
              <img src="https://github.com/ApatShe.png?size=100" width="100px;" alt="ApatShe"/><br />
              <sub><b>ApatShe</b></sub>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/svemyh">
              <img src="https://github.com/svemyh.png?size=100" width="100px;" alt="svemyh"/><br />
              <sub><b>svemyh</b></sub>
            </a>
        </td>
    </tr>
</table>

## License

Distributed under the MIT License. See `LICENSE` for more information.
