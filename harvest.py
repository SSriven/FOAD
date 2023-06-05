import matplotlib.pyplot as plt
import numpy as np
import torch
from cifar10.vggnet import vgg_16
from load_dataset import load_cifar

model = vgg_16()
model.load_state_dict(torch.load('cifar10/VGG_16_model/basemodel_cifar10.pth.tar'))

masks_dict = {}

BatchSize = [16,32,64,128,256,512]
k = 0
for batchsize in BatchSize:
    train_loader,test_loader = load_cifar('cifar10',batchsize,batchsize)
    x = next(iter(train_loader))[0]
    print(x.shape)
    masks,cfg = model.channel_selected(x,n=2,s=0)
    masks_dict["B"+str(batchsize)] = masks
    # masks_dict[str(k)] = masks
    # k+=1

x = masks_dict.keys()
y = masks_dict.keys()
print(masks_dict.keys())

# list1 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 25, 26, 27, 28, 31, 32, 35, 38, 39, 41, 42, 43, 44, 50, 51, 52, 53, 54, 56, 57, 59, 61, 64, 65, 66, 67, 68, 69, 75, 76, 81, 82, 84, 85, 86, 88, 90, 93, 98, 103, 113, 119, 125]
# list2 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 25, 26, 27, 28, 32, 34, 35, 37, 38, 42, 43, 46, 50, 51, 52, 53, 54, 56, 57, 65, 66, 68, 75, 76, 78, 81, 82, 86, 88, 90, 93, 95, 98, 103, 104, 106, 113, 115, 125]

# a = list(set(list1) & set(list2))
# b = list(set(list1) | set(list2))
# print(a)
# print(b)
# print(len(a) / len(b))

for index in range(13):

    harvest = np.zeros((len(BatchSize),len(BatchSize)))
    i = 0
    j = 0

    for key1 in masks_dict.keys():
        j = 0
        for key2 in masks_dict.keys():
            masks1 = masks_dict.get(key1)
            masks2 = masks_dict.get(key2)

            # for i in len(masks1):
            layer1 = masks1[index]
            layer2 = masks2[index]

            a = len(list(set(layer1) & set(layer2)))
            b = len(list(set(layer1) | set(layer2)))

            harvest[i,j] = round(a / b,2)

            j += 1
        i += 1

    print(harvest)

    # vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
    #               "potato", "wheat", "barley"]
    # farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
    #            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

    # harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
    #                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
    #                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
    #                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
    #                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
    #                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
    #                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

    plt.xticks(np.arange(len(x)), labels=x, 
                        rotation=45, rotation_mode="anchor", ha="right")
    plt.yticks(np.arange(len(y)), labels=y)    
    # plt.title("Harvest of local farmers (in tons/year)")

    for i in range(len(y)):
        for j in range(len(x)):
            text = plt.text(j, i, harvest[i, j], ha="center", va="center", color="black")

    plt.imshow(harvest,cmap=plt.cm.Set3)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('layer' + str(index) + '.png')
    plt.show()
