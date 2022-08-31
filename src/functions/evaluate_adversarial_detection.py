def evaluate_adversarial_detection(dataset,labelset,network,dataset_name,t=0.01,evaluate=False) -> float:
    #Import custom functions
    from img_median_smoothing import median_smoothing
    from img_bit_reduction import bit_reduction
    import torch.nn.functional as F
    import torch
    #Define variables
    differences = []
    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0

    #Define bit reduction fot the corresponding datasets
    if (dataset_name == "CIFAR"):
        color_depth=4
    else:
        color_depth=1
    #Prepare dataloader
    for i, (data, target) in enumerate(zip(dataset, labelset), 0):
        if (dataset_name == "MNIST"):
            data = data.unsqueeze(dim=1)
        if (dataset_name == "CIFAR"):
            data = data.reshape(1, 3, 28, 28)
        with torch.no_grad():
            output = network(data)
            #Output median smoothed ensemble
            median_smoothed_batch = torch.clone(data)
            median_smoothed_batch = median_smoothing(median_smoothed_batch,dataset_name,kernel_size=2)
            output_ms = network(median_smoothed_batch)
            #Output bit reduced ensemble
            bit_reduced_batch = torch.clone(data)
            bit_reduced_batch = bit_reduction(bit_reduced_batch, dataset_name, bit=color_depth)
            output_br = network(bit_reduced_batch)
            #Test accuracy for further calculations (not implemented)
            pred = output.data.max(1, keepdim=True)[1]
            pred_ms = output_ms.data.max(1, keepdim=True)[1]
            pred_br = output_br.data.max(1, keepdim=True)[1]
            
            #Average of three networks. Alternative is to use majority voting using output.data.max(1)[1] for each output and write it to an array to get argmax (it's not needer here but can be used for code extensions)
            output_ens = (output + output_ms + output_br)/3
            pred_ens = output_ens.data.max(1, keepdim=True)[1]

            softmax_baseline = F.softmax(output, dim=1)
            softmax_median_smooth = F.softmax(output_ms, dim=1)
            softmax_bitreduction = F.softmax(output_br, dim=1)

            for x in range(len(pred)):

                #Calculate distance between baseline and rectificated softmax outputs
                difference_ms = abs(
                    (softmax_baseline[x])-(softmax_median_smooth[x]))
                difference_br = abs(
                    (softmax_baseline[x])-(softmax_bitreduction[x]))

                distance_list = [max(difference_br).item(),
                                 max(difference_ms).item()]

                difference_total = max(distance_list)

                differences.append(difference_total)
                #Evaluate t results
                if (evaluate):
                    if ((difference_total > t) and (i <= (len(dataset)/2)-1)):
                        true_positive += 1
                    if ((difference_total > t) and (i > (len(dataset)/2)-1)):
                        false_positive += 1
                    if ((difference_total < t) and (i <= (len(dataset)/2)-1)):
                        false_negative += 1
                    if ((difference_total < t) and (i > (len(dataset)/2)-1)):
                        true_negative += 1
    if (evaluate):
        print("True Positives:", true_positive)
        print("False Positives:", false_positive)
        print("True Negatives:", true_negative)
        print("False Negatives:", false_negative)

        print("Accuracy:", (true_positive+true_negative) /
              (true_positive+true_negative+false_negative+false_positive))
    else:
        print("Calculated train dataset distances")
    a = [0]*int((len(dataset)/2))
    b = [1]*int((len(dataset)/2))
    c = b + a
    return(differences, c)
