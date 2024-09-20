# Episodic Memory Model

This repository, **Episodic_Memory_Model**, is based on the [FlexModEHC](https://github.com/dmcnamee/FlexModEHC) repository developed by Dan McNamee and co-authors for their paper:

> **Flexible modulation of sequence generation in the entorhinal–hippocampal system**  
> Daniel C. McNamee, Kimberly L. Stachenfeld, Matthew M. Botvinick & Samuel J. Gershman  
> *Nature Neuroscience*, 24, 851–862 (2021)

This model aims to simulate and explore episodic memory, focusing on how it can be used for both recall and future-oriented cognitive processes like inference, mental simulation, and imagination.

## Overview

The **Episodic Memory Model** expands on the principles set forth in FlexModEHC, providing a framework to study not only the recall of episodic memories but also their role in generating future thinking. This model is structured to allow for modular analyses of episodic memory dynamics across different environments and datasets.

## Repository Structure

### Inheritance Chain

The model is organized through a chain of inheritance, where different components build on one another in a hierarchical manner:

1. **Environment_Episodic**: The base component, responsible for setting up the context in which the episodic memory processes occur. The primary environment used is `Episodic_environment`, with other variants available for specific cases.

2. **Generator_Episodic**: This component draws from the environment, setting up the sequence of states that are used in the memory model.

3. **Propagator_Episodic**: It extends the functionality of the generator by propagating sequences through the model, establishing connections between different memory elements.

4. **Sampler_Episodic**: The final component in the chain, the sampler utilizes the propagated information to sample memory states, simulating recall or imagination processes.

### Key Files

- **Analysis Files (`Analysis_*.py`)**: These files leverage the inheritance chain to implement and explore different functionalities of the model, including various types of memory simulations and analyses.

- **Figures (`Figures/`)**: Contains key figures generated during analysis, showcasing the model's performance and results.

### Data Sources

The model works with two primary datasets:

1. **Synthetic Data**: Designed to serve as a controlled ground truth, this dataset allows for the assessment of model biases and validation of core functionalities.

2. **Sherlock Recall Data**: A real-world dataset from Janice Chen's 2016 study, which provides empirical evidence on episodic memory recall. This dataset is used to validate the model against human memory processes.

## Getting Started

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/LubnaSAP/Episodic_Memory_Model.git
    cd Episodic_Memory_Model
    ```

2. **Install Dependencies**: Make sure to install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Analyses**: Use the `Analysis_*.py` scripts to run various analyses. For example:
    ```bash
    python Analysis_Episodic_Recall.py
    ```

4. **Explore Results**: Check the `Figures/` directory for visualizations generated during the analysis.

## Citing This Work

If you use this repository in your research, please cite the original paper:

McNamee, D. C., Stachenfeld, K. L., Botvinick, M. M., & Gershman, S. J. (2021). Flexible modulation of sequence generation in the entorhinal–hippocampal system. *Nature Neuroscience, 24*, 851–862.

## Acknowledgments

This repository builds upon the foundational work of the FlexModEHC model, special tahanks to Dan for making their code and insights publicly available.. Aleix Alcaceres (GitHub: [aleixalcacer](https://github.com/aleixalcacer)) initially started working on this model. 
Thanks to Raphael Kaplan & Dan McNamee for their ideas, inputd, contribution and guidance.


--- 
