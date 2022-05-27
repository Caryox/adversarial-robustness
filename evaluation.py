from collections import OrderedDict

import torch
from torch import nn, optim

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *


def eval_step(engine, batch): # create default evaluator for doctests
    return batch


default_evaluator = Engine(eval_step) # create default optimizer for doctests


param_tensor = torch.zeros([1], requires_grad=True)         # create default trainer for doctests
default_optimizer = torch.optim.SGD([param_tensor], lr=0.1) # as handlers could be attached to the trainer,
                                                            # each test must define his own trainer using `.. testsetup:`

def get_default_trainer():                # create default model for doctests

    def train_step(engine, batch):
        return batch

    return Engine(train_step)

default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(4, 2)),
    ('fc', nn.Linear(2, 1))
]))

manual_seed(666)

class evaluation:

    def __init__ (self,y_pred,y_true):
        self.y_pred = y_pred
        self.y_true = y_true
    
    def accuracy (self):
        metric = Accuracy() #Accuracy Function
        metric.attach(default_evaluator, "accuracy")
        state = default_evaluator.run([[self.y_pred, self.y_true]])
        print('Accuracy: ',state.metrics["accuracy"])

    def recall (self):
        metric = Recall(average=False) #Recall Function
        metric.attach(default_evaluator, "recall")
        state = default_evaluator.run([[self.y_pred, self.y_true]])
        print('Recall: ',state.metrics["recall"])

    def precision (self):
        metric = Precision(average=False) #Precision Function
        metric.attach(default_evaluator, "precision")
        state = default_evaluator.run([[self.y_pred, self.y_true]])
        print('Precision: ',state.metrics["precision"])
        
    def f1 (self):
        precision = Precision(average=False) #F1 Function
        recall = Recall(average=False)  
        F1 = precision * recall * 2 / (precision + recall + 1e-20)
        F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)

    def classification_report (self):
        metric = ClassificationReport(output_dict=True)
        metric.attach(default_evaluator, "cr")
        state = default_evaluator.run([[self.y_pred, self.y_true]])
        print('Classes: ',state.metrics["cr"].keys())
        print('0: ',state.metrics["cr"]["0"])
        print('1: ',state.metrics["cr"]["1"])
        print('2: ',state.metrics["cr"]["2"])
        print('macro avg: ',state.metrics["cr"]["macro avg"])
    
    #def auroc (self):
    #    roc_auc = ROC_AUC() #The ``output_transform`` arg of the metric can be used to perform a sigmoid on the ``y_pred``.
    #    roc_auc.attach(default_evaluator, 'roc_auc')
    #    state = default_evaluator.run([[self.y_pred, self.y_true]])
    #    print(state.metrics['roc_auc'])



a_true = torch.tensor([2, 0, 2, 1, 0, 1])          #Multiclass Input Tensors Example
a_pred = torch.tensor([
    [0.0266, 0.1719, 0.3055],
    [0.6886, 0.3978, 0.8176],
    [0.9230, 0.0197, 0.8395],
    [0.1785, 0.2670, 0.6084],
    [0.8448, 0.7177, 0.7288],
    [0.7748, 0.9542, 0.8573],
])


a = evaluation(a_pred, a_true)


a.accuracy()
a.classification_report()
a.precision()
