---
layout: post
comments: true
title: Tutorial 2. Training the Machine Learning Model
---

### Learning Objectives
* Use Scikit-learn for simple classification tasks
* Use RDKit for molecular representations, such as Morgan Fingerprint
* Use the ROC curve to represent model accuracy

## Definition of Classification vs. Regression Problems
Classification and regression are two fundamental types of supervised machine learning tasks. Classification involves predicting a discrete label or category. For example, identifying if an email is "spam" or "not spam" or determining whether a tumor is "benign" or "malignant." Regression, on the other hand, predicts continuous values, such as forecasting house prices or estimating a person's weight based on their height.

## Examples of Classification vs. Regression Tasks
Classification Example: Determining if a molecule is cancerous or non-cancerous.
Regression Example: Predicting the boiling point of a chemical compound based on its structure.
This is a Classification Task
In our case, distinguishing between cancerous and non-cancerous PAH molecules is a classification task. The model learns to assign a molecule to one of these two categories based on its features.

## Commonly Used Classification Algorithms
Logistic Regression: This algorithm models the probability that an input belongs to a particular category. It uses a logistic function to squeeze predicted values between 0 and 1, making it effective for binary classification problems. To learn more, visit [Logistic Regression - Scikit-Learn](:https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html)

Random Forest: This is an ensemble method that builds multiple decision trees during training and combines their outputs for improved accuracy and robustness. It reduces overfitting and handles large datasets efficiently. More details can be found here: [Random Forest - Scikit-Learn](:https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html)


Support Vector Machine (SVM): SVMs classify data by finding the optimal hyperplane that separates different classes with the largest margin. It’s particularly effective for high-dimensional spaces. Learn more at [Support Vector Machines - Scikit-Learn](:https://scikit-learn.org/1.5/modules/svm.html)


These algorithms are widely used in classification tasks and can be explored further through these resources.

## How to Train a Machine Learning Model
Training a machine learning (ML) model requires dividing your data into two essential subsets: the training set and the test set. The training set is used to teach the model, while the test set evaluates its generalization to unseen data. A common and effective split ratio is 80% for training and 20% for testing. For classification tasks, metrics such as accuracy, true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN) offer insights into the model’s performance, with the ROC curve providing a deeper understanding of trade-offs between sensitivity and specificity. Comparing training set performance to test set performance reveals if the model is overfitting or underfitting.
This tutorial builds on the previous one, where we curated molecular data, preparing it for machine learning (ML) model training. Now, with the data processed and ready, we can use ML techniques to classify molecular structures and predict their properties.

## Code:
&nbsp;  
### Set up


<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
import os    
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import operator
import numpy as np
import sklearn.preprocessing
import sklearn.utils
from sklearn.decomposition import PCA 
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, accuracy_score
import sklearn.metrics as sklm
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import pandas as pd
from rdkit.Chem import AllChem
import pubchempy as pcp
from rdkit import Chem



fs = 10 # font size
fs_label = 10 # tick label size
fs_lgd = 10 # legend font size
ss = 20 # symbol size
ts = 3 # tick size
slw = 1 # symbol line width
framelw = 1 # line width of frame
lw = 2 # line width of the bar box
rc('axes', linewidth=framelw)
plt.rcParams.update({
    "text.usetex": False,
    "font.weight":"bold",
    "axes.labelweight":"bold",
    "font.size":fs,
    'pdf.fonttype':'truetype'
})
plt.rcParams['mathtext.fontset']='stix'
{% endhighlight %}
</pre>
* These code blocks set up the basic infrastructure for the model, including importing the necessary programming libraries and defining parameters for the data tables.

&nbsp;  
&nbsp;  
### Pre-processing the data set
&nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
datapath = 'PAH/' # path to your data folder
filein_test= os.path.join(datapath,'testset_0.ds') # read in the CSV file containing the features. This file is just for example
filein_train= os.path.join(datapath,'trainset_0.ds')
# The dataframe for molecule name and classis
df_test = pd.read_csv(filein_test, sep=" ",  header=None, names=['molecule', 'cancerous'])
df_train = pd.read_csv(filein_train, sep=" ",  header=None, names=['molecule', 'cancerous'])
{% endhighlight %}
</pre>
* This code block processes the data (different cancerous and uncancerous molecules) listed in a specified data folder
  &nbsp;
  &nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
df_test
{% endhighlight %}
</pre>
* This code block runs the function df_test, outputting data from a folder into a table with parameters defined by the previous code blocks, as shown below:
  
![image](https://github.com/user-attachments/assets/29aa6c34-9130-4316-925d-41c21857218b)

* This data is from the test set
* The testing set consists of data that is not used during training. It acts as a reference to evaluate the model's performance. These molecules have known cancerous or non-cancerous properties, allowing you to measure how accurately the model can make predictions based on the patterns it learned. By keeping this data separate, we prevent the model from simply memorizing the training data, ensuring it can generalize to new, unseen examples.

  &nbsp;
  &nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
df_train
{% endhighlight %}
</pre>


![image](https://github.com/user-attachments/assets/93b00e86-d1e6-421a-a88f-1a746b70c2b1)

* This data is from the training set
* The training set contains data that the model uses to learn patterns and relationships. In this case, these are molecules with unknown cancerous properties. The model analyzes their structures and attempts to find patterns that correlate with being cancerous or not. The goal is to expose the model to diverse data so it can generalize and make predictions about unseen molecules.


&nbsp;  
&nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
def getSMILES(df):
    mols=df['molecule'].values
    smiles_list = []
    for mol in mols:
        # Get rid of the ".ct" suffix
        # Search Pubchem by the compound name
        results = pcp.get_compounds(mol[:-3], 'name')
        smiles = ""
        if len(results) > 0:
            # Get the SMILES string of the compound
            smiles = results[0].isomeric_smiles
            smiles_list.append(smiles)
            print(mol[:-3],smiles)
        else:
            smiles_list.append(smiles)
            print(mol[:-3],'molecule not found in PubChem')
    df['SMILES'] = smiles_list

mymol = pcp.get_compounds('naphthalene', 'name', record_type='3d')[0]

mydict=mymol.to_dict(properties=['atoms'])

mydict['atoms']
{% endhighlight %}
</pre>
* This code block uses the external library PubChemPy, a tool to access PubChem, the world’s largest collection of freely accessible chemical information.
* The name of each molecule will be found in the PubChem database, and a SMILES string will then be generated based on its molecular structure.
* The data is converted into a dictionary using the to_dict function, resulting in an output of the molecule (in this case, naphthalene) as a list of its elements and their positions, a partial representation seen below:

  ![image](https://github.com/user-attachments/assets/72262dff-79f7-4cf7-9e8b-e34d205a03d0)


   &nbsp;
  &nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
getSMILES(df_train)
{% endhighlight %}
</pre>
* This function converts all the molecules in the training set into SMILES strings.

![image](https://github.com/user-attachments/assets/aed90f93-e0a1-4612-bd4c-31a5b5a64207)

&nbsp;  

&nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
getSMILES(df_test)
{% endhighlight %}
</pre>
* This function converts all the molecules in the testing set into SMILES strings

  ![image](https://github.com/user-attachments/assets/7b1c33da-e344-49c4-87d3-cfe487565905)

&nbsp;  


&nbsp;  
&nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
fpgen = AllChem.GetMorganGenerator(radius=2)
mol = Chem.MolFromSmiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
fp = fpgen.GetFingerprintAsNumPy(mol)

for i in fp:
    print(i)
{% endhighlight %}
</pre>
* This code block utilizes AllChem's Morgan Fingerprint generator to take the structural data processed from the SMILES strings. It then denotes the molecules as an array of integers
&nbsp;  
&nbsp;  

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
def getData(df):
    fpgen = AllChem.GetMorganGenerator(radius=2)
    MFP_list = []
    for smiles in df['SMILES'].values:
        mol = Chem.MolFromSmiles(smiles)
        MFP = fpgen.GetFingerprintAsNumPy(mol)
        MFP_list.append(MFP)
    X = np.array(MFP_list)

    y_list = []
    for y in df['cancerous']:
        if y == 1:
            y_list.append(1)
        else:
            y_list.append(0)
    y = np.array(y_list)
    return X,y

X_test,y_test = getData(df_test)

print(X_test)
{% endhighlight %}
</pre>

* This code block converts the SMILES strings from the test set into an array which represents its Morgan Fingerprint. It also converts its cancerous or uncancerous labels into a binary format
* The output can be seen below:

![image](https://github.com/user-attachments/assets/bc2ac740-4d0f-493f-9670-3843c31b956d)


&nbsp;  
&nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
X_train,y_train = getData(df_train)
{% endhighlight %}
</pre>

&nbsp;  
&nbsp;  
### Train a SVM and get the ROC curve
&nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
{% endhighlight %}
</pre>
* This code block imports the SVM module from scikit-learn which is then used to train the SVM classifier using the training data X_train and y_train
&nbsp;
&nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
clf.predict(X_test)
{% endhighlight %}
</pre>
* This function uses the SVM to predict the cancerous labels of the Molecules in X test

![image](https://github.com/user-attachments/assets/db7c4102-c3d6-4811-841f-ea25ab4ec0c9)

&nbsp;  
&nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
y_test
{% endhighlight %}
</pre>
* y_test is the actual set of cancerous labels that apply to the molecules in the test set

![image](https://github.com/user-attachments/assets/6cb39244-23af-4bee-9981-791c8ae58bc4)

* A comparison between the predicted labels and the actual labels shows that the SVM is not fully accurate
&nbsp;
&nbsp;  

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
from sklearn.metrics import RocCurveDisplay
svc_disp = RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.show()
{% endhighlight %}
</pre>
* This code block will plot the ROC (Receiver operating characteristic) Curve of the previously trained classifier. Evaluating a machine learning model is essential to ensure it performs well on unseen data. Metrics like accuracy are useful but may not tell the full story. The ROC (Receiver Operating Characteristic) curve and its AUC (Area Under the Curve) provide a deeper insight, showing the trade-off between true positive and false positive rates. A significant gap between training and test set performance often signals overfitting, where the model memorizes training data instead of generalizing patterns. To address this, techniques like cross-validation, regularization, or adding more diverse training data can help improve the model's accuracy and reliability for computational chemistry applications.

![image](https://github.com/user-attachments/assets/293d25ac-d197-41a3-baac-b8525f6d518b)

* The positive rate and false positive rate are plotted along the two different axes, the area under the curve denoting its accuracy

&nbsp;  
&nbsp;  
### Train a random forest and get the ROC curve
&nbsp;  
&nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)
svc_disp.plot(ax=ax, alpha=0.8)
plt.show()
{% endhighlight %}
</pre>

&nbsp;  
&nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
from sklearn.linear_model import LogisticRegression

clf_lg = LogisticRegression(random_state=0).fit(X_train, y_train)
clf.predict(X_test)
{% endhighlight %}
</pre>
*
![image](https://github.com/user-attachments/assets/9a84c109-270a-4afb-9a0b-2e60b00c8374)

*This shows the accuracy of the model when Random Forest Classifier and Logistic Regression is used

&nbsp;  
&nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>


*
![image](https://github.com/user-attachments/assets/3a89a637-4c25-4c9e-98fe-565e0b0760d1)


&nbsp;  
&nbsp;  
### Upsampling
&nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
from sklearn.utils import resample,shuffle
df_0 = df_train[df_train['cancerous'] == -1]
df_1 = df_train[df_train['cancerous'] == 1]

len(df_0), len(df_1)
{% endhighlight %}
</pre>
![image](https://github.com/user-attachments/assets/177b530f-6adf-467a-a635-9f07d6907c9e)

&nbsp;  
&nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
df_0_upsampled = resample(df_0,random_state=42,n_samples=50,replace=True)

len(df_0_upsampled)
{% endhighlight %}
</pre>
![image](https://github.com/user-attachments/assets/958a4f8c-61f4-4e2f-bdee-48bc35bf04d6)

&nbsp;  
&nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    },
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
df_0_upsampled
{% endhighlight %}
</pre>
![image](https://github.com/user-attachments/assets/1ff23392-638a-47a1-a0f3-b22b6d5dcf81)
![image](https://github.com/user-attachments/assets/2cd41a27-0901-4c77-be3e-98b33127db9d)
![image](https://github.com/user-attachments/assets/65bb7e04-330c-4d88-bdb9-e2260ab3fab1)
&nbsp;  
&nbsp;  
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>

<script type="text/x-thebe-config">
  {
    requestKernel: true,
    binderOptions: {
      repo: "matplotlib/ipympl",
      ref: "0.6.1",
      repoProvider: "github",
    }, 
  }
</script>
<script src="https://unpkg.com/thebe@latest/lib/index.js"></script>

<button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
  Activate
</button>
<script>
var bootstrapThebe = function() {
    thebelab.bootstrap();
}
document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
</script>

<pre data-executable="true" data-language="python">
{% highlight python %}
df_upsampled = pd.concat([df_0_upsampled,df_1])

X_train_up,y_train_up = getData(df_upsampled)

clf_lg = LogisticRegression(random_state=0).fit(X_train_up, y_train_up)
clf.predict(X_test)
{% endhighlight %}
</pre>
![image](https://github.com/user-attachments/assets/7cdff1f0-2d56-4767-8da3-87503313822d)

* This shows the accuracy when Upsampling is used with Logistic Regression
&nbsp;  















