import sys
sys.path.append('./utils')
import foolbox
import device

def attack(model, attack):
    """ Returns the attack object and the foolbox model. Use a loop to call this function for each attack. """
    fmodel = foolbox.models.PyTorchModel(model, bounds=(-1, 1), device=device.device)
    if(attack == "FGSM"):
        attack = foolbox.attacks.FGSM()
    elif(attack == "L2DeepFool"):
        attack = foolbox.attacks.L2DeepFoolAttack()
    elif(attack == "C&W"):
        attack = foolbox.attacks.L2CarliniWagnerAttack()
    elif(attack == "PGD"):
        attack = foolbox.attacks.PGD()
    
    return attack, fmodel

def evaluation(model, input, label):
    """ Returns the accuracy of the model on the input data. """
    return foolbox.utils.accuracy(model, input, label) * 100