
*************************************** Tier One ****************************************************

Pre-requisites: 

- Access to eICU Collaborative Research Database v2.0.
- Generate patients_pneumonia.csv  --- filtered data from 'patients.csv' containing only pneumonia patient data.
- Generate new_lab.csv from lab.csv with only records of potassium levels.

Data selection and preprocessing:

1) Run readandfind.py.
2) Run K_vals.py.
3) Run past_hist.py.


Deep Learning codes:

-Use Tier1_prob_death.py for calculating probability of death.

-Use Tier1_prob_venti.py for calculating probability of requiring mechanical ventilation.

(To regenerate our results use load_model from keras to load our pre-trained models from 'Pre-trained models' folder, instead of training)   

