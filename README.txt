Capstone Project User Manual

This package contains three versions of our brain-to-text project: baseline, diphones, and reduced phoneme models. Each version follows the same workflow for setup, training, and evaluation. Because the project is compute-intensive, experiments were run on Virginia Tech ARC (or a similar high-performance computing environment).

Project folders:

* src/capstone-proj-b2t-baseline/
* src/capstone-proj-b2t-diphones/
* src/capstone-proj-b2t-reduced/

Installation:
Open one of the three experiment folders and follow its README for:

* environment setup
* dataset download and placement
* language model download and setup
* conda environment creation

The language model only needs to be installed and set up once in the b2txt_lm environment and can be reused across all three experiment versions.

Conda environment usage:

* b2txt: training for baseline and reduced phoneme models
* b2txt_d: training for diphone models
* b2txt_lm: evaluation and language model decoding for all versions

Make sure the conda environment names match the names expected by the provided SLURM scripts before submitting jobs.

How to use:

1. Choose one experiment version.
2. Complete the setup steps in that folder.
3. Go to the model_training directory.

You may either:

* train from scratch using the provided training script, or
* copy the matching pretrained model from /trained_models into model_training/trained_models and run evaluation only

4. Submit the provided training and/or evaluation .sh SLURM scripts on ARC (or equivalent infrastructure).

Demo:
Run one training job and one evaluation job for any one version, or run evaluation only using a provided pretrained model. Once the job(s) complete successfully, the demo is complete.

Additional folders:

* src/evaluation_metrics/ contains predicted sentence outputs and JSON evaluation results.
* src/trained_models/ contains saved checkpoints, training curves, and the training curve generation script.
