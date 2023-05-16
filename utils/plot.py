import matplotlib.pyplot as plt
import os
import torch

def plot_loss_acc(args, train_loss, test_loss, train_acc, test_acc):
    if not os.path.exists(args.loss_path):
        os.makedirs(args.loss_path)
    # loss
    plt.plot(train_loss, 'r')
    plt.plot(test_loss, 'b')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    filename = os.path.join(args.loss_path, 'loss.jpg')
    plt.savefig(filename)
    plt.show()
    plt.close()
    # accuracy
    plt.plot(train_acc,'r')
    plt.plot(test_acc,'b')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(args.loss_path, 'accuracy.jpg'))
    plt.show()
    plt.close()

def plot_shapley(value, path, str="", vmax=None):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if vmax == None:
        vmax = max(value.max(), -value.min())
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.matshow(value, cmap=plt.get_cmap('bwr'), vmax=vmax, vmin=vmin)
    fig.colorbar(im)

    plt.title(f"Shapley values for sample {str}")
    plt.savefig(path+'.jpg', transparent=True, bbox_inches="tight")
    plt.show()
    plt.close()

