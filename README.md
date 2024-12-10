# ez-Quench
A simple model for jet quenching in Heavy-Ion Collisions

## Description and Usage
The ezQuench package consists of a set of python routines that fit a simple model for jet energy-loss (jet-quenching) in heavy-ion collisions measured by experiments at the Relativistic Heavy-Ion Collider (RHIC) and the Large Hadron Collider (LHC).  The routines currently perform a set of Bayesian calibrations to jet nuclear modification factors measured by the ATLAS collaboration for Pb-Pb collisions at the nucleon-nucleon center of mass energy of 5.02 TeV.  This software package serves as the companion code to [list arXiv reference] and can be used to generate all results and figures found therein.

## Directory Structure
src - all python source code
h5  - input data files
fig - figures created by Plotting routines

## Analysis Source Code (src/)
The following 5 analysis routines will generate all physics results:
1. ezQ_ppJet.py - use scipy.optimize to fit ATLAS pp-Jet cross-sections data from https://doi.org/10.17182/hepdata.84819 table 4 
2. ezQ_fitRAA.py - use emcee package to perform Bayesian calibration of simple jet-quench model to ATLAS central jet-RAA data from https://doi.org/10.1103/PhysRevC.107.054909
3. ezQ_trentoPaths.py - reads in TRENTO 2D grid for heavy-ion geometry and calculate path length for dijet pairs
4. ezQ_fitTrentoRAA.py - use emcee to perform Bayesian calibration for jet-RAA over all centralities
5. ezQ_helper.py - log-likelihood and other fit functions for the simple jet-quenching model

## Plotting Source Code (src/)
The following 5 files are used to plot figures in the paper
1. ezQ_plotRAA.py - plot results of Bayesian fit to central jet-RAA
2. ezQ_plotQuench.py - plot jet-quench distributions for simple model
3. ezQ_plotCorner.py - make corner plots for Bayesian fit to central jet-RAA
4. ezQ_plotTrentoRAA.py - plot results of Bayesian fit to jet-RAA for all collision centralities
5. ezQ_plotPPP.py - plot joint probability distributions for Peele's Pertinent Puzzle

## Data Files (h5/)
ATLAS_data.h5 - contains all jet-RAA data and errors for ATLAS
ALICE_CMS_data.h5 - contains all jet-RAA data and errors for ATLAS and CMS
trento_PbPb_10k.h5 - contains initial geometry data 10,000 TRENTO minimum bias PbPB events

## Release
This code has been approved for release by LLNL as LLNL-CODE-2001499, and is released under the MIT License.

