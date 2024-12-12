# Human Insights Driven Latent Space for Different Driving Perspectives: A Unified Encoder for Efficient Multi-Task Inference

## Description
This repository contains the code and resources accompanying the paper  
**"Human Insights Driven Latent Space for Different Driving Perspectives: A Unified Encoder for Efficient Multi-Task Inference"**.

It provides implementation details, scripts, and instructions for setting up the environment, preparing datasets, and running evaluations and demos related to the research.

## RDI Method: EDC(s)
*(If this repository does not involve EDCs, you may remove this section. Otherwise, list relevant EDCs and their details.)*

| EDCs | PDE | Internal Article | Current State | Main File | Main Contributor | Corresponding EDC |
|------|-----|-----------------|---------------|-----------|-----------------|------------------|
| ...  | ... | ...             | ...           | ...       | ...             | ...              |

## Requirements
- Python 3.8
- PyTorch 1.10.1 (CUDA 11.3 build)
- Detectron2-v0.6

Additional requirements and dependencies are described in [INSTALL.md](INSTALL.md).

## Installation
Follow the steps in [INSTALL.md](INSTALL.md) to set up the environment and install all necessary packages and dependencies.

Example:
```
python -m pip install -r /path/to/requirements.txt
```

## Usage
This repository provides scripts and functionalities to run evaluations and demos. For detailed commands and workflow instructions, refer to the following sections.

### Evaluation
See [GETTING_STARTED.md](GETTING_STARTED.md) for evaluation commands and guidelines.

### Demo
For running inference demos and showcasing the model’s capabilities, refer to [demo/README.md](demo/README.md).

## Documentation
Comprehensive documentation, including an overview of the code structure, methodology, and configuration details, can be found in the associated documentation files and comments within the source code.

## Contributing
If you would like to contribute:
- Report issues or suggest improvements in the issue tracker.
- Submit merge/pull requests following the project’s contributing guidelines.

## License
This code is for internal use and research purposes only. Consult the project maintainers for any usage outside the intended scope.

## Contact
For questions or inquiries, please contact the maintainer:
- Name: [Maintainer Name]
- Email: [Maintainer Email]

## Acknowledgements
We thank all collaborators, researchers, and projects that have influenced this work, including the community around Detectron2 and related tools.

## References
If this work is useful in your research, please consider starring the repository and citing it:

```
@misc{nguyen2024humaninsightsdrivenlatent,
      title={Human Insights Driven Latent Space for Different Driving Perspectives: A Unified Encoder for Efficient Multi-Task Inference}, 
      author={Huy-Dung Nguyen, Anass Bairouk, Mirjana Maras, Wei Xiao, Tsun-Hsuan Wang, Patrick Chareyre, Ramin Hasani, Marc Blanchon and Daniela Rus},
      year={2024},
      eprint={2409.10095},
      url={https://arxiv.org/abs/2409.10095}}
```