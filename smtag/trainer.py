from torch import nn, optim
from random import shuffle
import logging

class Trainer:

    def __init__(self, minibatches, validation_minibatches, model):

        self.model = model
        logging.info(f"model is: \n{model}")
        self.minibatches = minibatches
        self.validation = validation_minibatches

    def train(self, opt):
        self.learning_rate = opt['learning_rate']
        self.epochs = opt['epochs']
        loss_fn = nn.SmoothL1Loss() #good alternative is torch.nn.BCE()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for e in range(self.epochs):
            shuffle(self.minibatches)
            for m in self.minibatches:
                input, target = m.input, m.output
                optimizer.zero_grad()
                prediction = self.model(input)
                loss = loss_fn(prediction, target)
                print(f"loss={loss}")
                loss.backward()
                optimizer.step()
