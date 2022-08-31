#Import runtime libraries
import os
import sys

#Append needed funcion/module paths
sys.path.append('./utils')
sys.path.append('./src/Models')
sys.path.append('./src/functions')
sys.path.append('./src/results')
sys.path.append('./src')
sys.path.append('./src/APE_GAN')
sys.path.append('./src/IMG_REC_ENS')

#Support function for clearing terminal output
def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

#Project main
def project_main ():
    print("##########")
    print("Evaluation of the Effectiveness using Ensemble Adversarial Training in Combination with Conventional Input Rectification and APE-GAN Architecture in Combination with Few2Decide against Adversarial Perturbation Attacks​")
    print("##########")
    print("\n")
    print("1. APE-GAN Architecture in combination with Few2Decide")
    print("2. Ensemble adversarial training in combination with conventional input rectification")
    user_input = int(input("Defense-Model: "))
    if user_input == 1:
        cls()
        from APEGAN_main import main
    if user_input == 2:
        cls()
        from img_rec_ens_main import core_main
#Run main
project_main()