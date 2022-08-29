import torch
import numpy as np
import foolbox as fb
from torch.autograd import Variable

def generate_pertubated_examples(model,test_loader,attack_method=None,dataset_name="MNIST",n=100,epsilon=[0.1]):

    device = torch.device("cpu")
    adversarial_data = []
    legitimate_data = []
    legitimate_label = []
    adversarial_orignal_labels = []
    collected_ad=0
    collected_leg=0
    model = model.eval()
    print("Run attack: {}".format(attack_method))
    print("Run attack with epsilon: {}".format(epsilon))
    print("Generating {} samples for legitimate and adversarial".format(n))
    for i, data in enumerate(test_loader,0):
        from torch.autograd import Variable
        input, label = data
        input, label = Variable(input.to(device)), Variable(label.to(device))
            # Generate adversarial dataset for attack
        if (dataset_name=="CIFAR"):
            fmodel = fb.PyTorchModel(model, bounds=(-1.989473819732666, 2.130864143371582), device="cpu") #CIFAR
        else:
            fmodel = fb.PyTorchModel(model, bounds=(-0.4242129623889923, 2.821486711502075), device="cpu") #MNIST
        attack = attack_method
            #attack = fb.attacks.L2CarliniWagnerAttack()
        epsilons = epsilon #, 0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0

        raw_advs, advs, is_adv = attack(fmodel, input, label, epsilons=epsilons)
        adversarial_indexes = np.where((is_adv[0].cpu().view(-1).numpy()).astype(int)==1)
        legitimate_indexes = np.where((is_adv[0].cpu().view(-1).numpy()).astype(int)==0)
        #print (adversarial_indexes)
        for adversarials in range(len(adversarial_indexes[0])):
            if collected_ad==n:
                break
            #print(adversarials)
            adversarial_orignal_labels.append(label[adversarial_indexes[0][adversarials]])
            adversarial_data.append(advs[0][adversarial_indexes[0][adversarials]])
            collected_ad+=1
        for legitimates in range(len(legitimate_indexes[0])):
            if collected_leg==n:
                break
            legitimate_label.append(label[legitimate_indexes[0][legitimates]])
            legitimate_data.append(input[legitimate_indexes[0][legitimates]])
            collected_leg+=1
        if (collected_leg==n and collected_ad==n):
            break
    if(collected_leg!=n):
        print("Not enough legitimate samples")
    if(collected_ad!=n):
        print("Not enough adversarial samples")
    
    print("Adversarial samples: ",len(adversarial_data))
    print("Legitimate samples: ",len(legitimate_data))
    if (dataset_name=="CIFAR"):
        return (adversarial_data,torch.from_numpy(np.array(adversarial_orignal_labels)),legitimate_data,torch.from_numpy(np.array(legitimate_label)))
    else:
        return (torch.cat(adversarial_data).unsqueeze(dim=1),torch.from_numpy(np.array(adversarial_orignal_labels)),torch.cat(legitimate_data).unsqueeze(dim=1),torch.from_numpy(np.array(legitimate_label)))   #dim1 = mnist