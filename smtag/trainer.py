from torch import nn, optim
from random import shuffle
import logging
#to start tensorboard server: tensorboard --logdir runs
from tensorboardX import SummaryWriter
#from visualdl import LogWriter
from viz import Show

class Trainer:

    def __init__(self, minibatches, validation_minibatches, model):

        self.model = model
        #visualization of the training with tensorboardX
        self.writer = SummaryWriter()
        self.minibatches = minibatches
        self.validation_minibatches = validation_minibatches
        self.loss_fn = nn.SmoothL1Loss() #good alternative is torch.nn.BCE()
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, opt):
    
        self.learning_rate = opt['learning_rate']
        self.epochs = opt['epochs']
        self.optimizer.lr = self.learning_rate
        for e in range(self.epochs):
            shuffle(self.minibatches)
            for m in self.minibatches:
                input, target = m.input, m.output
                self.optimizer.zero_grad()
                prediction = self.model(input)
                loss = self.loss_fn(prediction, target)
                loss.backward()
                self.optimizer.step()
            validation_loss = self.validate()
            self.writer.add_scalars('data/loss', {'train':loss, 'valid':validation_loss}, e)
            Show.example(self.model, self.validation_minibatches)
            print(f"epoch {e}\ttrain_loss={loss}\tvalid_loss={validation_loss}\n")
        self.writer.close()

    def validate(self):
        self.model.eval()
        losses = 0
        for m in self.validation_minibatches:
            input, target = m.input, m.output
            prediction = self.model(input)
            loss = self.loss_fn(prediction, target)
            losses += loss
        self.model.train()
        avg_loss = losses / len(self.validation_minibatches)
        return avg_loss
