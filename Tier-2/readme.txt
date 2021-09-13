
****************************************************** Tier Two **************************************************


Pre-requisites:

-Access to MIMIC III Clinical Database.
-bloodvals_24.csv generated using blood-gas-first-day.sql written by Alistair Johnson and Tom Pollard ** (available on https://doi.org/10.5281/zenodo.821872).
-GPU with CUDA support.

Data selection and pre-processing:

1) Download MIMIC III Waveform Database Matched Subset.
2) Generate diagnosis_pneumonia.csv by filtering DIAGNOSIS_ICD.csv for pneumonia patients.
3) Run Read_database.py.
4) Run Data_preprocessing.py.

Deep learning codes:

- Use Tier2_death_wo_mechvent.py for calculating probability of death .

- Use Tier2_prob_venti.py for calculating probability of requiring mechanical ventilation.


(To regenerate our results use load_model from keras to load our pre-trained models from 'Pre-trained models' folder, instead of training)    







**
    Johnson, Alistair EW, David J. Stone, Leo A. Celi, and Tom J. Pollard. "The MIMIC Code Repository: enabling reproducibility in critical care research." Journal of the American Medical Informatics Association (2017): ocx084.

