def calculate_threshold_t(differences_test,adversarial_map,correction_t=1.0,print_output=True,is_evaluate=False,t=None,is_paper=False) -> float:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    difference_map = zip(differences_test,adversarial_map)
    adversarial_distances = []
    non_adversarial_distances = []
    for x in range (len(adversarial_map)):
        if (adversarial_map[x]==1):
            adversarial_distances.append(differences_test[x])
        if (adversarial_map[x]==0):
            non_adversarial_distances.append(differences_test[x])
    
    print("Max adversarial distance: ",max(adversarial_distances))
    print("Min adversarial distance: ",min(adversarial_distances))
    print("Mean adversarial distances: ",np.array(adversarial_distances).mean())
    print("##########")
    print(max(non_adversarial_distances))
    print(min(non_adversarial_distances))
    print(np.array(non_adversarial_distances).mean())
    t_distance=(min(adversarial_distances)+max(non_adversarial_distances))/2
    t_legitimate_mean=np.array(non_adversarial_distances).mean()
    t_adaptive=(t_distance+t_legitimate_mean)/2
    print("Distance t legitimate mean:",t_legitimate_mean)
    print("Distance t min-max:",t_distance)
    print("Distance t conservative mean:",t_adaptive)
    # Should t be more conservative or aggressive. Higher t means more conservative, resulting in less True Positives but less False Positives. True Negative Detection is rising with higher t and falling True Positive

    non_adversarial_distances=np.array(non_adversarial_distances)
    adversarial_distances=np.array(adversarial_distances)
    # Measure Quality of the three t_calc Models
    true_t_distance = np.count_nonzero(non_adversarial_distances[non_adversarial_distances<=t_distance])+np.count_nonzero(adversarial_distances[adversarial_distances>=t_distance])
    false_t_distance = np.count_nonzero(non_adversarial_distances[non_adversarial_distances>=t_distance])+np.count_nonzero(adversarial_distances[adversarial_distances<=t_distance])
    true_t_adaptive = np.count_nonzero(non_adversarial_distances[non_adversarial_distances<=t_adaptive])+np.count_nonzero(adversarial_distances[adversarial_distances>=t_adaptive])
    false_t_adaptive = np.count_nonzero(non_adversarial_distances[non_adversarial_distances>=t_adaptive])+np.count_nonzero(adversarial_distances[adversarial_distances<=t_adaptive])
    true_t_legitimate_mean = np.count_nonzero(non_adversarial_distances[non_adversarial_distances<=t_legitimate_mean])+np.count_nonzero(adversarial_distances[adversarial_distances>=t_legitimate_mean])
    false_t_legitimate_mean = np.count_nonzero(non_adversarial_distances[non_adversarial_distances>=t_legitimate_mean])+np.count_nonzero(adversarial_distances[adversarial_distances<=t_legitimate_mean])
    score_peak_distance = true_t_distance-false_t_distance
    score_adaptive = true_t_adaptive-false_t_adaptive
    score_legitimate_mean = true_t_legitimate_mean-false_t_legitimate_mean
    max_score = max(score_peak_distance,score_adaptive,score_legitimate_mean)
    min_score = min(score_peak_distance,score_adaptive,score_legitimate_mean)
    print("Score legitimate mean:",score_legitimate_mean)
    print("Score min-max:",score_peak_distance)
    print("Score conservative mean:",score_adaptive)
    print("##########")
    print("Max Score:",max_score)
    print("Min Score:",min_score)
    print("Mean Score:",(score_peak_distance+score_adaptive+score_legitimate_mean)/3)
    print("##########")
    if (score_peak_distance==max_score):
        print ("Evaluated min-max mean calculation as best threshold for t")
        t=t_distance
    if (score_adaptive==max_score):
        print ("Evaluated conservative mean calculation as best threshold for t")
        t=t_adaptive
    if (score_legitimate_mean==max_score):
        print ("Evaluated legitimate mean calculation as best threshold for t")
        t=t_legitimate_mean
    print("True T Min-Max: ",true_t_distance-false_t_distance)
    print("True T Conservative: ",true_t_adaptive-false_t_adaptive)
    print("True T Legitimate: ",true_t_legitimate_mean-false_t_legitimate_mean)
    #If print_output, show plt graphic
    if (print_output):
        limit=max(len(adversarial_distances),len(non_adversarial_distances))
        plt.figure(figsize=(30,10))
        plt.hlines(y=t,xmin=0,xmax=limit,color='green',linestyle='dashed')
        plt.xlabel("Image samples")
        plt.ylabel("L1-Distance")
        plt.plot(adversarial_distances,c="red")
        plt.plot(non_adversarial_distances,c="blue")
        plt.legend(["Adversarial","Legitimate","Threshold"])
        plt.grid()
        plt.show()
    print("Final threshold t: ",t)
    print("Legitimate Samples ",len(non_adversarial_distances))
    print("Adversarial Samples ",len(adversarial_distances))
    print("True Negatives ",np.count_nonzero(non_adversarial_distances[non_adversarial_distances<=t]))
    print("False Positives ",np.count_nonzero(non_adversarial_distances[non_adversarial_distances>=t]))
    print("True Positives ",np.count_nonzero(adversarial_distances[adversarial_distances>=t]))
    print("False Negatives",np.count_nonzero(adversarial_distances[adversarial_distances<=t]))
    print("##########TRAIN RESULTS##########")
    print("Detection Ratio TP: ",(100/len(adversarial_distances))*np.count_nonzero(adversarial_distances[adversarial_distances>=t]))
    print("Detection Ratio FN: ",(100/len(adversarial_distances))*np.count_nonzero(adversarial_distances[adversarial_distances<=t]))
    print("Detection Ratio TN: ",(100/len(non_adversarial_distances))*np.count_nonzero(non_adversarial_distances[non_adversarial_distances<=t]))
    print("Detection Ratio FP: ",(100/len(non_adversarial_distances))*np.count_nonzero(non_adversarial_distances[non_adversarial_distances>=t]))
    return t