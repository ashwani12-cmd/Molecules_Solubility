#!/usr/bin/env python
# coding: utf-8

# In[200]:


import pandas as pd
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
df=pd.read_csv("Solubility_Data.csv")
Molecules=[]
for molecule in df.SMILES:
    mol = Chem.MolFromSmiles(molecule)
    Molecules.append(mol)    
    
MolLog=[]
Molwt=[]
NumRotatableBonds=[]
HeavyAtomsCount=[]
AromaticAtomCount=[]
AromaticProportion=[]
for i,item in enumerate(Molecules):
    append_MolLogP = Descriptors.MolLogP(Molecules[i])
    MolLog.append(append_MolLogP)
    append_Molwt = Descriptors.MolWt(Molecules[i])
    Molwt.append(append_Molwt)
    append_NumRotatableBonds = Descriptors.NumRotatableBonds(Molecules[i])
    NumRotatableBonds.append(append_NumRotatableBonds)
    append_HeavyAtomCount=Descriptors.HeavyAtomCount(Molecules[i])
    HeavyAtomsCount.append(append_HeavyAtomCount)
    count=len(Molecules[i].GetAromaticAtoms())
    AromaticAtomCount.append(count)
    count=0
    
for i,item in enumerate(Molecules):
    var=AromaticAtomCount[i]/HeavyAtomsCount[i]
    AromaticProportion.append(var)
    
descriptors=np.array([MolLog,Molwt,NumRotatableBonds,AromaticProportion])
descriptors=descriptors.transpose()
columnNames=["MolLogP","MolWt","NumRotatableBonds","Aromatic_Proportion"]   
final_descriptors = pd.DataFrame(data=descriptors,columns=columnNames)

X=final_descriptors
Y=df.iloc[:,2]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
reg = LinearRegression()
reg.fit(X_train, Y_train)
Y_predict=reg.predict(X_test)
m=mean_squared_error(Y_test, Y_predict)

print('Mean squared error (MSE) of the testing set: %.2f'%m)
print('Coefficient of determination (R^2) for the testing set: %.2f'% r2_score(Y_test, Y_predict))
print("The Accuracy of the Regression Model is: %0.2f" % ((cross_val_score(reg, X, Y, cv=5)).mean()))

