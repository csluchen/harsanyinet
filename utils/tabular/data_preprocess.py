import os
import numpy as np
import json
import requests
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

ROOT_DIR = 'data/TabularRawData'

CENSUS_DATASET = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
)

YEAST_DATASET = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.names"
)
COMMERCIAL_DATASET = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/00326/TV_News_Channel_Commercial_Detection_Dataset.zip",
)

def download_data(path='data/TabularRawData/Census', urls=CENSUS_DATASET):
    if not os.path.exists(path):
        os.makedirs(path)

    for url in urls:
        name = os.path.basename(url)
        response = requests.get(url)
        with open(os.path.join(path, name), 'wb') as f:
            f.write(response.content)


        if os.path.splitext(name)[-1]== '.zip':
            import zipfile
            z = zipfile.ZipFile(os.path.join(path,"TV_News_Channel_Commercial_Detection_Dataset.zip"))
            z.extractall(path)
            os.remove(os.path.join(path,"TV_News_Channel_Commercial_Detection_Dataset.zip"))

 
def dump_json(root,attributes, target, X_train, X_test, MEAN, STD):
    meta = {
            "attributes": attributes,
            "target":target,
            "n_attributes": len(attributes),
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "X_mean_original" : list(MEAN),
            "X_std_original":list(STD)}
    with open(os.path.join(root,'info.json'), 'w') as f:
        json.dump(meta, f, indent=2)



def read_data(dataset, download=True):
    if os.path.exists(os.path.join(ROOT_DIR,dataset)):
       download=False

    if dataset == 'Census':
        if download:
            download_data(path=os.path.join(ROOT_DIR,dataset), urls=CENSUS_DATASET)
        names = ['age',
                 'workclass',
                 'fnlwgt',
                 'education',
                 'education-num',
                 'marital-status',
                 'occupation',
                 'relationship',
                 'race',
                 'sex',
                 'capital-gain',
                 'capital-loss',
                 'hours-per-week',
                 'native-country',
                 'income']
        data = pd.read_csv(os.path.join(ROOT_DIR,'Census/adult.data'),names = names)
        data_test = pd.read_csv(os.path.join(ROOT_DIR,'Census/adult.test'),names=names)
        data_test = data_test.iloc[1:]
        data = data.drop(['fnlwgt','education'],axis=1) #remove repetible information 
        data_test =data_test.drop(['fnlwgt','education'],axis=1)
        print(data.head())
        ### clean data
        data['occupation'] = data['occupation'].replace('?',np.nan)
        data['workclass'] = data['workclass'].replace('?',np.nan)
        data['native-country'] = data['native-country'].replace('?', np.nan)
        data.dropna(how = 'any', inplace=True)

        data_test['occupation'] = data_test['occupation'].replace('?',np.nan)
        data_test['workclass'] = data_test['workclass'].replace('?',np.nan)
        data_test['native-country'] = data_test['native-country'].replace('?', np.nan)
        data_test.dropna(how = 'any', inplace=True)
        data_test['age'] = data_test['age'].astype(int)
        ### mapping the data into numerical data using map function
        #data['income'] = data['income'].map({'<=50K':0, '>50K':1}).astype(int)
        #data_test['income'] = data_test['income'].map({'<=50K':0, '>50K':1}).astype(int)

        CatagoricalFeature = ['workclass','marital-status','occupation','relationship','race','sex','native-country','income']
        target = 'income'    
    elif dataset == 'Yeast':
        if download:
            download_data(path=os.path.join(ROOT_DIR,dataset), urls=YEAST_DATASET)
        names = ['Sequence Name','mcg','gvh','alm','mit','erl','pox','vac','nuc','class']
        data = pd.read_csv(os.path.join(ROOT_DIR,'Yeast/yeast.data'),header=0,names=names, delim_whitespace=True)
        data = data.drop(['Sequence Name'],axis=1)
        print(data.head())
        CatagoricalFeature = ['class']
        target ='class' 
        data, data_test = train_test_split(data, test_size=0.2)    
    
    elif dataset == 'Commercial':
        if download:
            download_data(path = os.path.join(ROOT_DIR, dataset), urls=COMMERCIAL_DATASET)
        files = ['BBC.txt','CNN.txt','CNNIBN.txt','NDTV.txt','TIMESNOW.txt']
        names = ['Shot Length',
                 'Motion Distribution',
                 'Frame Difference Distribution',
                 'Short time energy',
                 'ZCR',
                 'Spectral Centroid',
                 'Spectral Rol off',
                 'Spectral Flux',
                 'Fundamental Frequency',
                 'Motion Distribution',
                 'Edge change Ratio',
                 'class'
                 ]
         
        dic_names = {'1':'Shot Length',
                     '2':'Motion Distribution',
                     '4':'Frame Difference Distribution',
                     '6':'Short time energy',
                     '8':'ZCR',
                     '10':'Spectral Centroid',
                     '12':'Spectral Rol off',
                     '14':'Spectral Flux',
                     '16':'Fundamental Frequency',
                     '4124':'Edge change Ratio',
                 }
        
        dic = {}
        for name in names:
            dic[name] = []

        for file in files:
            f = open(os.path.join(ROOT_DIR,dataset,file),'r')
            Lines = f.readlines()
            for line in Lines:
                line = line.split()
                dic['class'].append(line[0])
                ind = 0
                for l in line:
                    l = l.split(':')
                    if len(l)>1:
                        if l[0] in dic_names.keys():
                            dic[dic_names[l[0]]].append(float(l[1]))
        data = pd.DataFrame.from_dict(dic)
        print(data.head())
        CatagoricalFeature = ['class']
        target ='class' 
        data, data_test = train_test_split(data, test_size=0.2)    


    LE = LabelEncoder()
    for column in CatagoricalFeature:
        data[column] = LE.fit_transform(data[column])
        data_test[column] = LE.fit_transform(data_test[column])

    print(data.head())
    train_label = data[target]
    data = data.drop([target],axis=1)
    Y_train = train_label.to_numpy()
    X_train = data.to_numpy()
    test_label = data_test[target]
    data_test = data_test.drop([target],axis=1)
    Y_test = test_label.to_numpy()
    X_test = data_test.to_numpy()

    MEAN = data.mean().to_numpy()
    STD = data.std().to_numpy()
    root = f'data/{dataset}'
    if not os.path.exists(root):
        os.mkdir(root)

    dump_json(root, data.columns.to_list(),target, X_train, X_test,MEAN, STD)
    np.save(os.path.join(root, 'X_train_raw.npy'), X_train)
    np.save(os.path.join(root, 'Y_train.npy'), Y_train)
    np.save(os.path.join(root, 'X_test_raw.npy'), X_test)
    np.save(os.path.join(root, 'Y_test.npy'), Y_test)

    #### standarlize the data, test data share the same mean and std with training data
    X_train = (data-data.mean())/data.std()
    X_test = (data_test - data.mean())/data.std()

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    np.save(os.path.join(root, 'X_train.npy'), X_train)
    np.save(os.path.join(root, 'X_test.npy'), X_test)

if __name__ == '__main__':
    read_data('Census')        
    read_data('Yeast')
    read_data('Commercial')
