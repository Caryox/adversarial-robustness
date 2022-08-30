def evaluate_attack(loader,network,dataset_name,base_accuracy,device,is_attack=False,attack=None,epsilon=None,t=[0.0],median_kernel=2) -> float:
    import foolbox as fb
    from test_ensemble import test
    from torch.autograd import Variable
    from img_median_smoothing import median_smoothing
    from img_bit_reduction import bit_reduction
    import time
    import torch
    import torch.nn.functional as F
    import numpy as np

    train_losses = []
    runs=0
    train_counter = []
    test_losses = []
    test_acc = []
    debug_output = 0
    test_loss = 0
    correct = 0
    correct_ms = 0
    correct_pool = 0
    correct_br = 0
    correct_ensemble = 0
    pertubated = 0
    pertubated_detected = 0
    differences=[]
    distance_l0 = 0
    distance_l1 = 0
    distance_l2 = 0
    distance_linf = 0
    adversarial_map = []
    baseline_threshold = 0
    false_positives = 0
    true_negatives = 0
    if (dataset_name == "CIFAR"):
        color_depth=4
    else:
        color_depth=1
    if (is_attack):
        print("Attack Mode!")
        model = network.eval()
        time.sleep(1)
        if  "MNIST" in str(loader.dataset):
            fmodel = fb.PyTorchModel(model, bounds=(-0.4242129623889923, 2.821486711502075), device="cpu")
        else:
            fmodel = fb.PyTorchModel(model, bounds=(-1.989473819732666, 2.130864143371582), device="cpu")
    
    for i, data in enumerate(loader,0):
        input, target = data
        input, target = Variable(input.to(device)), Variable(target.to(device))
        if (is_attack):
            raw_advs, advs, is_adv = attack(fmodel, input, target, epsilons=epsilon)
            data = advs[0]
            pertubated += np.count_nonzero(is_adv.cpu().view(-1).numpy())
            adversarial_map.append(is_adv[0].numpy().astype(int))
            # Output original ensemble
        else:
            data = input
        with torch.no_grad():
            output = network(data)

            # Output median smoothed ensemble
            median_smoothed_batch=torch.clone(data)
            median_smoothed_batch = median_smoothing(median_smoothed_batch,dataset_name,kernel_size=median_kernel)
            output_ms = network(median_smoothed_batch)

            # Output bit reduced ensemble
            bit_reduced_batch=torch.clone(data)
            bit_reduced_batch = bit_reduction(bit_reduced_batch, dataset_name, bit=color_depth) #CIFAR = 4
            output_br = network(bit_reduced_batch)

            # Losses for each ensemble
            test_loss += F.cross_entropy(output, target).item()

            # Test accuracy
            pred = output.data.max(1, keepdim=True)[1]
            pred_ms = output_ms.data.max(1, keepdim=True)[1]
            pred_br = output_br.data.max(1, keepdim=True)[1]

            output_ens = (output + output_ms + output_br)/3# Average of three networks. Alternative is to use majority voting using output.data.max(1)[1] for each output and write it to an array to get argmax
            pred_ens = output_ens.data.max(1, keepdim=True)[1]

            softmax_baseline= F.softmax(output, dim=1)
            softmax_median_smooth = F.softmax(output_ms, dim=1)
            softmax_bitreduction = F.softmax(output_br, dim=1)
            

            # Jury Vote for ensemble predictions
            jury_vote = []
            for x in range(len(pred)):
                votes = []
                first_vote = output.data.max(1, keepdim=True)[1][x].item()
                second_vote = output_ms.data.max(1, keepdim=True)[1][x].item()
                third_vote = output_br.data.max(1, keepdim=True)[1][x].item()
                votes.append(first_vote)
                votes.append(second_vote)
                votes.append(third_vote)
                jury_vote.append(np.bincount(votes).argmax())

            # Calculate distance between baseline and rectificated softmax outputs
                difference_ms=abs((softmax_baseline[x])-(softmax_median_smooth[x]))
                difference_br=abs((softmax_baseline[x])-(softmax_bitreduction[x]))


                distance_list=[max(difference_br).item(),max(difference_ms).item()]
                #difference_total = statistics.fmean(distance_list)
                difference_total = max(distance_list)
                #print("Max Tensor Distance:", difference_total)
                differences.append(difference_total)
                if (is_attack):
                    if ((difference_total > t) and ((is_adv[0].numpy().astype(int))[x] == 1)):
                            pertubated_detected += 1

                if(target[x].item()==jury_vote[x]):
                    correct_ensemble += 1
                else:
                    #print("Original target:",target[x].item(),"| Jury voted class:",jury_vote[x],"| Vanilla Ensemble Vote:" ,first_vote,"Median-Smoothed Ensemble Vote:",second_vote,"Bit-Reduced Ensemble Vote:",third_vote)
                    if (debug_output):
                        fig = plt.figure()
                        ax1 = fig.add_subplot(131)  # left side
                        ax2 = fig.add_subplot(132)  # right side
                        ax3 = fig.add_subplot(133)  # right side
                    
                        ax1.title.set_text("Original Image")
                        ax2.title.set_text("Median Smoothed Image")
                        ax3.title.set_text("Bit Reduced Image")

                        ax1.imshow(data[x][0], cmap='gray', interpolation='none')
                        ax2.imshow(median_smoothed_batch[x][0], cmap='gray', interpolation='none')
                        ax3.imshow(bit_reduced_batch[x][0], cmap='gray', interpolation='none')
                        fig.text(.5, .05, "Original Target: {}".format(target[x].item()), ha='center')
                        fig.set_figheight(5)
                        fig.set_figwidth(10)
                        plt.show()


            correct += pred.eq(target.data.view_as(pred)).sum()
            correct_ms += pred_ms.eq(target.data.view_as(pred_ms)).sum()
            correct_br += pred_br.eq(target.data.view_as(pred_br)).sum()
            correct_pool += pred_ens.eq(target.data.view_as(pred_ens)).sum()
            if (is_attack):
                distance_l0 += (fb.distances.l0(raw_advs[0], input)/32).mean().item()
                distance_l1 += (fb.distances.l0(advs[0], input)/32).mean().item()
                distance_l2 += (fb.distances.l2(raw_advs[0], input)/32).mean().item()
                distance_linf += (fb.distances.linf(raw_advs[0]/32, input)).mean().item()
            runs+=1
            
    test_loss /= len(loader.dataset)
    test_losses.append(test_loss)
    acc = 100. * correct / len(loader.dataset)
    test_acc.append(acc)
    
    print('\nTest set: Avg. loss baseline: {:.4f}, Accuracy: {}/{} ({:.0f}%), Accuracy_Median_Smooth: {}, Accuracy_Bit_Reduction: {}, Accuracy_Jury_Ensemble: {}, Accuracy_Pooled_Ensemble: {}\n'.format(
    test_loss, acc, len(loader.dataset),
    100. * correct / len(loader.dataset), 100. * correct_ms / len(loader.dataset), 100. * correct_br / len(loader.dataset), 100. * correct_ensemble / len(loader.dataset),100. * correct_pool / len(loader.dataset)))
    print('Attack Success (Pooled): {}'.format((100-(100/base_accuracy)-acc)/100))

    print('L0 Distance: {}'.format(distance_l0/runs))
    print('L2 Distance: {}'.format(distance_l2/runs))
    print('Linf Distance: {}'.format(distance_linf/runs))
    #return(differences,np.array(adversarial_map).flatten())
    