# It's not a real function but the core code for selecting specific netowrk features and usability functions



log_interval = 187 #60000/batch_size_train = 1875 => 10 Datapoints per Epoch => 187 Logs per Epoch
n_epochs = 20
learning_rate = 0.007
momentum = 0.5
load_data = 1
use_upgraded_net = 1
use_ensemble = 0
##########
save_net = 0
##########

if (use_upgraded_net == 1):
    network = simple_net_upgraded(1,10)
    network.apply(reset_weights)
    print ("Using Upgraded Net")
    if (use_ensemble == 1):
        network2 = simple_net_upgraded2(1,10)
        network3 = simple_net_upgraded3(1,10)
        
        network2.apply(reset_weights)
        network3.apply(reset_weights)

        print("Using Ensemble")
    
else:
    network = simple_net()
    network.apply(reset_weights)
    print ("Using Basic Net")


#device = "cuda:0"
#network.to(device)
if (use_ensemble == 0):
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
if (use_ensemble == 1):
    optimizer_ensemble = optim.SGD(list(network.parameters())+list(network2.parameters())+list(network3.parameters()), lr=learning_rate, momentum=momentum)
    print("Loaded Optimizer for Ensemble")
if (load_data and use_upgraded_net == 0):
    network.load_state_dict(torch.load('../src/results/model.pth'))
    print('Loaded model from file')
    optimizer.load_state_dict(torch.load('../src/results/optimizer.pth'))
    print('Loaded optimizer from basic model')
if (load_data and use_upgraded_net == 1):
    if (use_ensemble==1):
        network.load_state_dict(torch.load('../src/results/model_upgraded_ens.pth'))
        network2.load_state_dict(torch.load('../src/results/model2_upgraded_ens.pth'))
        network3.load_state_dict(torch.load('../src/results/model3_upgraded_ens.pth'))
        print('Loaded upgraded ensemble model from file')
        optimizer_ensemble.load_state_dict(torch.load('../src/results/optimizer_ensemble.pth'))
        print('Loaded ensemble optimizer from upgraded model')
    else:
        network.load_state_dict(torch.load('../src/results/model_upgraded.pth'))
        print('Loaded upgraded model from file')
        optimizer.load_state_dict(torch.load('../src/results/optimizer_upgraded.pth'))
        print('Loaded optimizer from upgraded model')
if (save_net==1):
    print('WARNING: Saving model to file!! Will overwrite existing file!!!')