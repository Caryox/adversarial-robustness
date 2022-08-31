#Running import of function inside of code if code is extended sometimes
#Train function
def train(network,optimizer,train_loader,dataset_name,epochs,adversarial_train=False,double_adv=False):
  import foolbox as fb
  import torch.nn.functional as F
  import torch.optim as optim
  network.train()
  train_losses = []
  train_counter = []
  if (adversarial_train):
    if (double_adv):
      epsilons = [0.3]
      print ("Using FGSM+PGD alterating adversarial train")
    else:
      epsilons = [0.05]
      print ("Using PGD+Normal alterating adversarial train")
  else:
    print("Using normal train")
  if (str(dataset_name)=="CIFAR"):
    log_interval=156 #50000/32/10 for each 10%
  else:
    log_interval=187 #60000/32/10 for each 10%
  print ("Train for {} epochs".format(epochs))
  for epoch in range(epochs):
    epoch+=1
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      if (adversarial_train):
        if (double_adv):
          if (str(dataset_name)=="CIFAR"):
            if (epoch%2==0): #For alternating train
              epsilons = [0.3]
              network.eval()
              fmodel = fb.PyTorchModel(network, bounds=(-1.989473819732666, 2.130864143371582), device="cpu") #Define bounds for pixel
              attack = fb.attacks.PGD()
              raw_advs, advs, is_adv = attack(fmodel, data, target, epsilons=epsilons)
              data = advs[0]
              network.train()
              optimizer.zero_grad()
            else:
              epsilons = [0.3]
              network.eval()
              fmodel = fb.PyTorchModel(network, bounds=(-1.989473819732666, 2.130864143371582), device="cpu") #Define bounds for pixel
              attack = fb.attacks.FGSM()
              raw_advs, advs, is_adv = attack(fmodel, data, target, epsilons=epsilons)
              data = advs[0]
              network.train()
              optimizer.zero_grad()
          else:
            if (epoch%2==0):
              epsilons = [0.3]
              network.eval()
              fmodel = fb.PyTorchModel(network, bounds=(-0.4242129623889923, 2.821486711502075), device="cpu") #Define bounds for pixel
              attack = fb.attacks.PGD()
              raw_advs, advs, is_adv = attack(fmodel, data, target, epsilons=epsilons)
              data = advs[0]
              network.train()
              optimizer.zero_grad()
            else:
              epsilons = [0.3]
              network.eval()
              fmodel = fb.PyTorchModel(network, bounds=(-0.4242129623889923, 2.821486711502075), device="cpu") #Define bounds for pixel
              attack = fb.attacks.FGSM()
              raw_advs, advs, is_adv = attack(fmodel, data, target, epsilons=epsilons)
              data = advs[0]
              network.train()
              optimizer.zero_grad()
        else:
          if (str(dataset_name)=="CIFAR"):
            if (epoch%2==0):
              epsilons = [0.3]
              network.eval()
              fmodel = fb.PyTorchModel(network, bounds=(-1.989473819732666, 2.130864143371582), device="cpu") #Define bounds for pixel
              attack = fb.attacks.PGD()
              raw_advs, advs, is_adv = attack(fmodel, data, target, epsilons=epsilons)
              data = advs[0]
              network.train()
              optimizer.zero_grad()
          else:
            if (epoch%2==0):
              epsilons = [0.3]
              network.eval()
              fmodel = fb.PyTorchModel(network, bounds=(-0.4242129623889923, 2.821486711502075), device="cpu") #Define bounds for pixel
              attack = fb.attacks.PGD()
              raw_advs, advs, is_adv = attack(fmodel, data, target, epsilons=epsilons)
              data = advs[0]
              network.train()
              optimizer.zero_grad()
      #Get model data
      output = network(data)
      loss = F.cross_entropy(output, target)
      loss.backward()
      optimizer.step()
      #Print outputs
      if (batch_idx % log_interval == 0) and (batch_idx != 0):
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
        #Not needed yet, but can be used for analyse train
        train_losses.append(loss.item())
        train_counter.append((int(batch_idx)*32) + ((int(epoch)-1)*int(len(train_loader.dataset))))