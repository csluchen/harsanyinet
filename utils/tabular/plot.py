import os
import matplotlib.pyplot as plt
import numpy as np

def RMSD(a,b):
    """calculate the root mean square deviation"""
    num_feature = a.shape[-1]
    error = np.sqrt(np.sum((a-b)**2)/num_feature)
    return error

def reject_outliers(data, m=2.):
    d = np.abs(data-np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def plot_convergence(TrueShap, EstShap, save_dir):
    '''function to plot the RMSD of the estimated shapley values vs. ground truth shapley values 
       using different  approximation methods. 
       :param TrueShap: np.darray, size (num_samples, num_features)
       :param EstShap: dict {str: np.darray}, keys = ['HarsanyiShapley','SamplingShapley','AntitheticalShapley',
       'KernelShapley','KernelPairShapley']
       :param save_dir: str
    '''
    
    num_samples = TrueShap.shape[0]
    num_features = TrueShap.shape[1]
    mean_value = {}  # get the mean of results on different samples
    std_value = {} # get the std of results on different samples
    
    for name, value in EstShap.items():
        # HarsanyiShapley only has one inference 
        if name =="HarsanyiShapley":
            error = []
            for num in range(num_samples):
                error.append(RMSD(TrueShap[num],value[num]))
            error = np.array(error)
            mean_value[name] = np.mean(error)
            std_value[name] = np.std(error)
        # other methods validate on different inference times
        
        else:
            num_sampling = value.shape[1]
            mean_value[name] = []
            std_value[name] =[]
            for i in range(num_sampling):
                error = []
                for num in range(num_samples):
                    error.append(RMSD(TrueShap[num],value[num][i]))
                error = np.array(error)
                error = reject_outliers(error)
                mean_value[name].append(np.mean(error))
                std_value[name].append(np.std(error))

    x = [1, 10, 32, 50, 100, 200, 500, 1000, 1500, 2000, 5000]
    
    y_1 = mean_value["HarsanyiShapley"]
    y_1_error = std_value["HarsanyiShapley"]

    y_2 = np.array(mean_value["SamplingShapley"])
    y_2_error = np.array(std_value["SamplingShapley"])

    y_3 = np.array(mean_value["KernelShapley"])
    y_3_error = np.array(std_value["KernelShapley"])
    
    y_4 = np.array(mean_value["KernelPairShapley"])
    y_4_error = np.array(std_value["KernelPairShapley"])
    
    y_5 = np.array(mean_value["AntitheticalShapley"])
    y_5_error = np.array(std_value["AntitheticalShapley"])
    
    plt.figure(figsize=(16,9))
    plt.plot(1, y_1,marker="*", markersize=60, markeredgecolor="red", markerfacecolor="red",clip_on=False, label='HarsanyiShapley')
    plt.fill_between(x, y_1- y_1_error, y_1+y_1_error, alpha=0.2)

                    
    plt.plot(x, y_2, label='SamplingShapley',linewidth=5)
    plt.fill_between(x, y_2- y_2_error, y_2+y_2_error, alpha=0.2)


    plt.plot(x, y_3, label='KernelShapley',linewidth=5)
    plt.fill_between(x, y_3- y_3_error, y_3+y_3_error, alpha=0.2)
    
    plt.plot(x[3:], y_4, label='KernelPairShapley',linewidth=5)
    plt.fill_between(x[3:], y_4- y_4_error, y_4+y_4_error, alpha=0.2)
    
    plt.plot(x, y_5, label='AntitheticalShapley',linewidth=5)
    plt.fill_between(x, y_5- y_5_error, y_5+y_5_error, alpha=0.2)
    
    plt.tick_params(width=5,length=10,which='both')
    plt.ylim(bottom=0)
    plt.xlim(left=1)
    #plt.xscale('log')
    plt.xticks(fontsize = 50)
    plt.yticks(fontsize = 50)
    plt.legend(loc='upper right',fontsize=40)
    plt.savefig(os.path.join(save_dir,'result.png'))
    plt.show()
    plt.close()



if __name__=="__main__":
    save_dir ='output/Census/layers3_hiddendim100_beta10_gamma100/ShapleyResult/'

    TrueShap = np.load(os.path.join(save_dir,'trueShapley.npy'))
    HarsanyiShapley  = np.load(os.path.join(save_dir, 'HarsanyiShapley.npy'))
    SamplingShapley = np.load(os.path.join(save_dir, 'SamplingShapley.npy'))
    KernelShapley = np.load(os.path.join(save_dir,'KernelShapley.npy'))
    KernelPairShapley = np.load(os.path.join(save_dir, 'KernelPairShapley.npy'))
    PermutationShapley = np.load(os.path.join(save_dir,'SamplingAntitheticalShapley.npy'))
    attr_dic= {'HarsanyiShapley':HarsanyiShapley,
                'SamplingShapley':SamplingShapley,
                'KernelShapley':KernelShapley,
                'KernelPairShapley':KernelPairShapley,
                'AntitheticalShapley':PermutationShapley}
    plot_convergence(TrueShap, attr_dic, save_dir)

