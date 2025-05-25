# STEM-Orchestrator : Enabling Real-Time Multimodal Data Acquisition in (Scanning transmission electron microscopes)STEM 

## [Link to Preprint - https://doi.org/10.31224/4645](https://engrxiv.org/preprint/view/4645/version/6316)

<img src="./assests/stemOrchestratorv2.png" width="1000" height="300" alt="STEM Orchestrator">

Why do we need STEM-Orchestrator?
- The various hardware components[eels detector, in-situ-holders, aberration corrector etc] of a STEM system often operate in isolation and lack seamless communication.
-  STEM-Orchestrator bridges this gap, enabling synchronized control and data sharing across devices—crucial for deploying complex machine learning workflows at the atomic scale and accelerating materials characterization.

Interested in contributing?
 - Please check CONTRIBUTING.md 
 - reach out


## Known issues - trouble-shooting:
### 1. Not seeing logging in action? 

- Chect if logger is set before importing utilities

 ```python
 # 1st set the logger
from stemOrchestrator.logging_config   import setup_logging
data_folder  = "."
out_path = data_folder
setup_logging(out_path=out_path) 

#Then import the utilities
from stemOrchestrator.acquisition import TFacquisition, DMacquisition

 ```


## Cite this work as
```

@Article{stemOrchestrator2025,
  author  = "Pratiush, Utkarsh and Houston, Austin and Longo, Paolo and Geurts, Remco and Kalinin, Sergei and Duscher, Gerd",
  title   = "stemOrchestrator: Enabling Seamless Hardware Control and High-Throughput Workflows on Electron Microscopes",
  journal = "engRxiv",
  year    = "2025",
  publisher = "Engineering Archive",
  doi     = "10.31224/4645",
  url     = "https://doi.org/10.31224/4645",
  keywords = "electron microscopy, instrumentation, automation"
}
```