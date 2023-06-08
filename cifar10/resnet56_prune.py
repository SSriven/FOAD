import os
import sys
sys.path.append(".")
import torch
from resnet56 import ResNet56,BasicBlock
from torchvision import datasets, transforms
from trainer import *
from load_dataset import load_cifar
from flops import print_model_param_flops,print_model_param_nums
from utils import init_log
import time
from matplotlib import pyplot as plt
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

BATCH_SIZZE = 128
epochs = 160
save = 'ResNet56_model'
t = 1
s = 0

train_loader,test_loader = load_cifar('cifar10',64,BATCH_SIZZE)

if not os.path.exists(save):
    os.makedirs(save)

logging = init_log(save)
_print = logging.info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet56().to(device)
model.load_state_dict(torch.load('ResNet56_model/basemodel_cifar10.pth.tar'),strict=False)
_, acc = test(model, device, test_loader)


_print("--------prune after----------")
x = next(iter(test_loader))[0].cuda()
masks,cfg = model.channel_selected(x,t=t,s=s)
np.savetxt(os.path.join(save, 'model_cfg(n='+str(t)+',s='+str(s)+').txt'),np.array(cfg),fmt='%d')
pruned_model = ResNet56(cfg=cfg).to(device)
_print(pruned_model)
print_model_param_nums(_print,pruned_model)
print_model_param_flops(_print,pruned_model,input_res=32,channels=3)


mask_id = 0
# fc_id = 1
last_mask = [i for i in range(3)]
for ((m1_name,m1_conv),(m2_name,m2_conv)) in zip(model.arc.items(),pruned_model.arc.items()):
    if isinstance(m1_conv,BasicBlock):
        hasshortcut = len(m1_conv.shortcut) == 3

        m2_conv.baseNet[0].weight.data = m1_conv.baseNet[0].weight.data[masks[mask_id]].clone()
        m2_conv.baseNet[0].weight.data = m2_conv.baseNet[0].weight.data[:,last_mask,:,:].clone()

        m2_conv.baseNet[1].weight.data = m1_conv.baseNet[1].weight.data[masks[mask_id]].clone()
        m2_conv.baseNet[1].bias.data = m1_conv.baseNet[1].bias.data[masks[mask_id]].clone()
        m2_conv.baseNet[1].running_mean = m1_conv.baseNet[1].running_mean[masks[mask_id]].clone()
        m2_conv.baseNet[1].running_var = m1_conv.baseNet[1].running_var[masks[mask_id]].clone()

        last_mask = masks[mask_id]
        mask_id += 1

        m2_conv.baseNet[3].weight.data = m1_conv.baseNet[3].weight.data[masks[mask_id]].clone()
        m2_conv.baseNet[3].weight.data = m2_conv.baseNet[3].weight.data[:,last_mask,:,:].clone()

        m2_conv.baseNet[4].weight.data = m1_conv.baseNet[4].weight.data[masks[mask_id]].clone()
        m2_conv.baseNet[4].bias.data = m1_conv.baseNet[4].bias.data[masks[mask_id]].clone()
        m2_conv.baseNet[4].running_mean = m1_conv.baseNet[4].running_mean[masks[mask_id]].clone()
        m2_conv.baseNet[4].running_var = m1_conv.baseNet[4].running_var[masks[mask_id]].clone()


        if hasshortcut:
            m2_conv.shortcut[0].weight.data = m1_conv.shortcut[0].weight.data[masks[mask_id]].clone()
            m2_conv.shortcut[0].weight.data = m2_conv.shortcut[0].weight.data[:,masks[mask_id-2],:,:].clone()

            m2_conv.shortcut[1].weight.data = m1_conv.shortcut[1].weight.data[masks[mask_id]].clone()
            m2_conv.shortcut[1].bias.data = m1_conv.shortcut[1].bias.data[masks[mask_id]].clone()
            m2_conv.shortcut[1].running_mean = m1_conv.shortcut[1].running_mean[masks[mask_id]].clone()
            m2_conv.shortcut[1].running_var = m1_conv.shortcut[1].running_var[masks[mask_id]].clone()

        last_mask = masks[mask_id]
        mask_id += 1

    else:
        # print(m2_name,m2_conv)
        m2_conv[0].weight.data = m1_conv[0].weight.data[masks[mask_id]].clone()
        m2_conv[0].weight.data = m2_conv[0].weight.data[:,last_mask,:,:].clone()

        m2_conv[1].weight.data = m1_conv[1].weight.data[masks[mask_id]].clone()
        m2_conv[1].bias.data = m1_conv[1].bias.data[masks[mask_id]].clone()
        m2_conv[1].running_mean = m1_conv[1].running_mean[masks[mask_id]].clone()
        m2_conv[1].running_var = m1_conv[1].running_var[masks[mask_id]].clone()

        last_mask = masks[mask_id]
        mask_id += 1

pruned_model.fc.weight.data = model.fc.weight.data[:,masks[mask_id-1]].clone()
pruned_model.fc.bias.data = model.fc.bias.data.clone()

_, acc = test(pruned_model, device, test_loader)
print_model_param_nums(_print,pruned_model)
print_model_param_flops(_print,pruned_model,input_res=32,channels=3)

_print("--------Retrain----------")
optimizer = torch.optim.SGD(pruned_model.parameters(),lr=0.1,weight_decay=1e-4,momentum=0.9)
best_prec1 = 0
start = time.time()
finetune_acc = []
finetune_loss = []
for epoch in range(1, epochs + 1):
    if epoch in [epochs*0.5,epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(pruned_model, device, train_loader, optimizer, epoch)
    _, acc = test(pruned_model, device, test_loader)
    log_msg = "epoch:{}/{} loss:{:.4f}  acc:{:.4f}".format(epoch,epochs,_,acc)
    finetune_acc.append(acc)
    finetune_loss.append(_)
    _print(log_msg)
    is_best = acc > best_prec1
    best_prec1 = max(acc, best_prec1)
    if is_best:
        torch.save(pruned_model.state_dict(), os.path.join(save, 'pruned_model_cifar10(n='+str(t)+'_'+str(s)+'_b'+str(BATCH_SIZZE)+').pth.tar'))
end = time.time()
_print("time:{}".format(end - start))
_print("best_acc:{:.4f}".format(best_prec1))


