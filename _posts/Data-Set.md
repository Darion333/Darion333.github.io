---
layout: post
comments: true
title: Tutorial 1. Configuring the Data Set
---

### Learning objectives
* Understand the basics of Chemoinformatics
* Interpret the raw data set
* Using API's like PubChem and RDKit to transfigure data

## Background

Identifying cancerous molecules is crucial for scientific research, especially when these molecules are used in various studies. Polycyclic aromatic hydrocarbons (PAHs) are molecules formed by multiple carbon rings. While some PAH molecules are carcinogenic, others are not, making distinguishing between them necessary in research settings.

Chemoinformatics is the scientific field that studies or predicts a molecule’s properties through informational techniques. Referencing a fundamental principle in chemistry, the similarity principle ,where molecules of similar structures exhibit similar properties, we can develop a machine learning model to classify PAH molecules. This model will compare the structures of unknown PAHs with those of molecules that have already been classified, predicting their carcinogenic potential.

To read more on Cheminformatics, this paper is a usefule resource: https://link.springer.com/chapter/10.1007/978-3-642-20844-7_12 

Additionally, We will use external libraries such as PubChemPy and RDKit. These tools will first represent the molecular structures as SMILES (Simplified Molecular Input Line Entry System) strings. The SMILES strings will then be converted into Morgan fingerprints, an array format that represents a molecule's structure in binary. This conversion is critical, as the molecular names stored as strings are insufficient for computational processing.

### Setting up the Data set

For our purposes, we will be using a set of PAH molecules downloadable from this link: https://www.researchgate.net/publication/234024134_PAHtar

Opening up these text files, the list should look something like this:

[PAH.tar.gz](https://github.com/user-attachments/files/17249526/PAH.tar.gz)










