import sys
sys.path.append('./utils')

import param, dataloader

import basic_classification_nn



#https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
from torch.autograd import Variable



def train(num_epochs, Net, loaders):
    
    Net.train()
        
    # Train the model
    total_step = len(loaders[0])
        
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(loaders[0],0):                                                            #, (images, labels) in enumerate(loaders['train']):
            
            inputs, labels = data
            # gives batch data, normalize x when iterate train_loader
            #b_x = Variable(images)   # batch x
            #b_y = Variable(labels)   # batch y

            # clear gradients for this training step   
            optimizer.zero_grad() 


            output = Net(inputs)                   #(b_x)[0]               
            loss = loss_func(output, labels)                  
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()          
            
            running_loss += loss.item()
            if i % 100 == 99:
                #print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Loss: {:.3f}' 
                #       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), running_loss))
                
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, running_loss))            
                
                running_loss = 0.0
                
                
                #loss_list.append(loss.item())
                #train_counter.append(i)
    
    print("Training abgeschlossen")
    
train(param.num_epochs, basic_classification_nn, dataloader.dataloader_MNIST[0])

