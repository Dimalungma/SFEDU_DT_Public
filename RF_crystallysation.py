import os
import numpy as np
import random
from matplotlib import pyplot as plt

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pandas as pd

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

import yaml
import encode_full_electronic_configuration

print("   ###   Libraries:")
print('   ---   sklearn:{}'.format(sklearn.__version__))
print('   ---   rdkit:{}'.format(rdkit.__version__))
np.random.seed(1)
random.seed(1)


def reg_stats(y_true,y_pred,scaler=None):
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  if scaler:
    y_true_unscaled = scaler.inverse_transform(y_true)
    y_pred_unscaled = scaler.inverse_transform(y_pred)
  r2 = sklearn.metrics.r2_score(y_true,y_pred)
  mae = sklearn.metrics.mean_absolute_error(y_true_unscaled, y_pred_unscaled)
  return r2,mae


def train(df, target, hparam):
    fontname='Arial' 
    outdir=os.getcwd()

    print("start training")


    if not os.path.exists("%s/scatter_plots"%(outdir)):
        os.makedirs("%s/scatter_plots"%(outdir))

    if not os.path.exists("%s/models"%(outdir)):
        os.makedirs("%s/models"%(outdir))

    if not os.path.exists("%s/predictions_rawdata"%(outdir)):
        os.makedirs("%s/predictions_rawdata"%(outdir))


    use_rdkit = hparam["use_rdkit"]   
    rdkit_l = hparam["rdkit_l"]
    rdkit_s = hparam["rdkit_s"]
    
    
    X = np.array(df.index.tolist())
    kf = KFold(n_splits=10,shuffle=True)
    kf.get_n_splits(X)
    counter=0  
    for train_index, test_index in kf.split(X):
        counter=counter+1
        
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(df[target].values.reshape(-1,1))  ###standard scaling of the output
        y_train, y_test = y[train_index], y[test_index]
   
        
        e1=encode_full_electronic_configuration.s1
        x_unscaled_feat_1=e1
        x_feat_1=x_unscaled_feat_1
 
        e2=encode_full_electronic_configuration.s2
        x_unscaled_feat_2=e2
        x_feat_2=x_unscaled_feat_2

        e3=encode_full_electronic_configuration.s3
        x_unscaled_feat_3=e3
        x_feat_3=x_unscaled_feat_3

        e4=encode_full_electronic_configuration.s4
        x_unscaled_feat_4=e4
        x_feat_4=x_unscaled_feat_4

        e5=encode_full_electronic_configuration.s5
        x_unscaled_feat_5=e5
        x_feat_5=x_unscaled_feat_5

        e6=encode_full_electronic_configuration.s6
        x_unscaled_feat_6=e6
        x_feat_6=x_unscaled_feat_6

        e7=encode_full_electronic_configuration.p2
        x_unscaled_feat_7=e7
        x_feat_7=x_unscaled_feat_7

        e8=encode_full_electronic_configuration.p3
        x_unscaled_feat_8=e8
        x_feat_8=x_unscaled_feat_8

        e9=encode_full_electronic_configuration.p4
        x_unscaled_feat_9=e9
        x_feat_9=x_unscaled_feat_9

        e10=encode_full_electronic_configuration.p5
        x_unscaled_feat_10=e10
        x_feat_10=x_unscaled_feat_7

        e11=encode_full_electronic_configuration.d3
        x_unscaled_feat_11=e11
        x_feat_11=x_unscaled_feat_11

        e12=encode_full_electronic_configuration.d4
        x_unscaled_feat_12=e12
        x_feat_12=x_unscaled_feat_12


        e13=encode_full_electronic_configuration.d5
        x_unscaled_feat_13=e13
        x_feat_13=x_unscaled_feat_13

        e14=encode_full_electronic_configuration.f4
        x_unscaled_feat_14=e14
        x_feat_14=x_unscaled_feat_14

    
        e15=encode_full_electronic_configuration.o
        x_unscaled_feat_15=e15
        x_feat_15=x_unscaled_feat_15


        if use_rdkit:


            x_unscaled_fp1 = np.array([Chem.RDKFingerprint(mol1, maxPath=rdkit_l, fpSize=rdkit_s) for mol1 in df['mol1'].tolist()]).astype(float)
            x_scaler_fp1 = StandardScaler()
            x_fp1 = x_scaler_fp1.fit_transform(x_unscaled_fp1)


            x_unscaled_fp2 = np.array([Chem.RDKFingerprint(mol2, maxPath=rdkit_l, fpSize=rdkit_s) for mol2 in df['mol2'].tolist()]).astype(float)
            x_scaler_fp2 = StandardScaler()
            x_fp2 = x_scaler_fp2.fit_transform(x_unscaled_fp2)


        x = np.hstack([x_fp1,x_fp2,x_feat_1,x_feat_2,x_feat_3,x_feat_4,x_feat_5,x_feat_6,x_feat_7,x_feat_8,x_feat_9,x_feat_10,x_feat_11,x_feat_12,x_feat_13,x_feat_14,x_feat_15])
        x_repeat = np.hstack([x_fp2,x_fp1,x_feat_1,x_feat_2,x_feat_3,x_feat_4,x_feat_5,x_feat_6,x_feat_7,x_feat_8,x_feat_9,x_feat_10,x_feat_11,x_feat_12,x_feat_13,x_feat_14,x_feat_15])
        x_unscaled = np.hstack([x_unscaled_fp1,x_unscaled_fp2,x_unscaled_feat_1,x_unscaled_feat_2,x_unscaled_feat_3,x_unscaled_feat_4,x_unscaled_feat_5,x_unscaled_feat_6,x_unscaled_feat_7,x_unscaled_feat_8,x_unscaled_feat_9,x_unscaled_feat_10,x_unscaled_feat_11,x_unscaled_feat_12,x_unscaled_feat_13,x_unscaled_feat_14,x_unscaled_feat_15])
        x_unscaled_repeat = np.hstack([x_unscaled_fp2,x_unscaled_fp1,x_unscaled_feat_1,x_unscaled_feat_2,x_unscaled_feat_3,x_unscaled_feat_4,x_unscaled_feat_5,x_unscaled_feat_6,x_unscaled_feat_7,x_unscaled_feat_8,x_unscaled_feat_9,x_unscaled_feat_10,x_unscaled_feat_11,x_unscaled_feat_12,x_unscaled_feat_13,x_unscaled_feat_14,x_unscaled_feat_15])

        
        x_train, x_test = x[train_index],x[test_index]
        x_unscaled_train, x_unscaled_test = x_unscaled[train_index], x_unscaled[test_index]
   
        r_train_index=[]
        for abcde in range(0,len(train_index)):
            if (df["nlinker"][train_index[abcde]])==2:
                r_train_index.append(train_index[abcde])   

        r_test_index=[]
        for abcde in range(0,len(test_index)):
            if (df["nlinker"][test_index[abcde]])==2:
                r_test_index.append(test_index[abcde])


        x_r_train, x_r_test = x_repeat[r_train_index],x_repeat[r_test_index] 
        x_r_unscaled_train, x_r_unscaled_test = x_unscaled_repeat[r_train_index], x_unscaled_repeat[r_test_index]
        y_r_train, y_r_test = y[r_train_index], y[r_test_index]


        x_train=np.vstack([x_train,x_r_train])  
        y_train=np.vstack([y_train,y_r_train])
        x_test=np.vstack([x_test,x_r_test])
        y_test=np.vstack([y_test,y_r_test])



        print("   ---   Training and test data dimensions:")   
        print(x_train.shape,x_test.shape,y_train.shape, y_test.shape)


     

        ##########################################
        # RandomForestRegressor model is initiated now
        ###############################################
        model =  RandomForestRegressor(max_depth=7)
        #fit the model
        model.fit(x_train,y_train.ravel())
    

        print("\n   ###   RandomForestRegressor:")
        y_pred_train = model.predict(x_train)
        r2_GBR_train,mae_GBR_train = reg_stats(y_train,y_pred_train,y_scaler)
        print("   ---   Training (r2, MAE): %.3f %.3f"%(r2_GBR_train,mae_GBR_train))
        y_pred_test = model.predict(x_test)
        r2_GBR_test,mae_GBR_test = reg_stats(y_test,y_pred_test,y_scaler)
        print("   ---   Testing (r2, MAE): %.3f %.3f"%(r2_GBR_test,mae_GBR_test))

        y_test_unscaled = y_scaler.inverse_transform(y_test)
        y_train_unscaled = y_scaler.inverse_transform(y_train)
        y_pred_test_unscaled = y_scaler.inverse_transform(y_pred_test)
        y_pred_train_unscaled = y_scaler.inverse_transform(y_pred_train)


        np.savetxt("%s/predictions_rawdata/y_real_test.txt"%(outdir), y_test_unscaled)
        np.savetxt("%s/predictions_rawdata/y_real_train.txt"%(outdir), y_train_unscaled)
        np.savetxt("%s/predictions_rawdata/y_RFR_test.txt"%(outdir), y_pred_test_unscaled)
        np.savetxt("%s/predictions_rawdata/y_RFR_train.txt"%(outdir), y_pred_train_unscaled)

        np.savetxt("./predictions_rawdata/y_real_"+str(counter)+"_test.txt", y_test_unscaled)
        np.savetxt("./predictions_rawdata/y_real_"+str(counter)+"_train.txt", y_train_unscaled)
        np.savetxt("./predictions_rawdata/y_RFR_"+str(counter)+"_test.txt", y_pred_test_unscaled)
        np.savetxt("./predictions_rawdata/y_RFR_"+str(counter)+"_train.txt", y_pred_train_unscaled)


        plt.figure()
        plt.scatter(y_pred_train_unscaled, y_train_unscaled, marker="o", c="C1", label="Training: r$^2$ = %.3f"%(r2_GBR_train))
        plt.scatter(y_pred_test_unscaled, y_test_unscaled, marker="o", c="C2", label="Testing: r$^2$ = %.3f"%(r2_GBR_test))
        plt.scatter(y_pred_train_unscaled, y_train_unscaled, marker="o", c="C1", label="Training: MAE = %.3f"%(mae_GBR_train))
        plt.scatter(y_pred_test_unscaled, y_test_unscaled, marker="o", c="C2", label="Testing: MAE = %.3f"%(mae_GBR_test))
        plt.plot(y_train_unscaled,y_train_unscaled)
        plt.title('RandomForestRegressor')
        plt.ylabel("Experimental Temperature [C]")
        plt.xlabel("Predicted Temperature [C]")
        plt.legend(loc="upper left")
        plt.savefig("%s/scatter_plots/full_data_RFR.png"%(outdir),dpi=300)
        plt.close()



target="crystalisaton"
df = pd.read_csv("full.csv")  ## The csv file containing the input output of the ML models


df['mol1']=df['linker1smi'].apply(lambda smi: Chem.MolFromSmiles(smi))   
df['mol2']=df['linker2smi'].apply(lambda smi: Chem.MolFromSmiles(smi))   


if os.path.exists("settings.yml"):
    user_settings = yaml.load(open("settings.yml","r"))
    hparam = yaml.load(open("settings.yml","r"))

train(df, target, hparam)  # call the function defined above to train the ML model


