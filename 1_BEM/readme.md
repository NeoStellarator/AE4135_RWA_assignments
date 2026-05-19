# Group 37 - Assignment #1 - BEM

Assignment choice: **Propeller**

Project members:
- Grzegorz Kamiński        5732727
- Alfonso Medina Marrero   5711274
- Kasper Weel              5523648

The project structure is as follows:
==== Folders
├───/data             : stores the airfoil data
├───/plots            : stores the plots
├───/results          : stores numerical results
├───/verification     : stores JavaProp verification data (ARAD6 propeller)
==== Main Physics Code
├───Annuli.py         : contains BEM code for a single annuli
├───Rotor.py          : discretizes blade and evaluates integrated properties
├───tip_correction.py : contains tip correction codes
├───multi_rotor_analysis.py : evaluates rotors for different conditions (e.g. varying J)
==== Additional Analysis
├───optimizer.py: perform optimization to maximize power generation
├───stag_pressure.py: perform analysis of the stagnation pressure distribution
==== Post-Processing Code
├───verification.ipynb : compares our BEM code with Ning and JavaProp
├───plotting_routines.py : contains plotting functions
└───plotting_sandbox.ipynb : creates results plot for the assingment questions (**run this one for all plots!!**)

Python Requirements:
- matplotlib
- pandas
- numpy
- scipy
- notebook
- jupyter
- juliacall (to verify against Ning Code)

Julia requirements
- CCBlade (to verify against Ning Code)
