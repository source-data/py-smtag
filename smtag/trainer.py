from torch import nn, optim
from random import shuffle
import logging
#to start tensorboard server: tensorboard --logdir runs
from tensorboardX import SummaryWriter
from viz import Show

class Trainer:

    def __init__(self, training_minibatches, validation_minibatches, model):

        self.model = model
        #visualization of the training with tensorboardX
        self.writer = SummaryWriter()
        self.minibatches = training_minibatches
        self.validation_minibatches = validation_minibatches
        self.loss_fn = nn.BCELoss() # nn.SmoothL1Loss() #

    def validate(self):
        self.model.eval()
        avg_loss = 0

        for m in self.validation_minibatches:
            input, target = m.input, m.output
            prediction = self.model(input)
            loss = self.loss_fn(prediction, target)
            avg_loss += loss

        self.model.train()
        avg_loss = avg_loss / len(self.validation_minibatches)

        return avg_loss

    def train(self, opt):

        self.learning_rate = opt['learning_rate']
        self.epochs = opt['epochs']
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        
        for e in range(self.epochs):
            shuffle(self.minibatches) # order of minibatches is randomized at every epoch
            avg_train_loss = 0 # loss averaged over all minibatches

            counter = 1
            for m in self.minibatches:
                input, target = m.input, m.output
                self.optimizer.zero_grad()
                prediction = self.model(input)
                loss = self.loss_fn(prediction, target)
                loss.backward()
                avg_train_loss += loss
                self.optimizer.step()
                print(f"\n\n\nepoch {e}\tminibatch #{counter}\tloss={loss}")
                Show.example(self.validation_minibatches, self.model)
                counter += 1

            avg_train_loss = avg_train_loss / self.minibatches.minibatch_number
            avg_validation_loss = self.validate() # the average loss over the validation minibatches
            print(f"\nepoch {e}\tavg_train_loss={avg_train_loss}\tavg_validation_loss={avg_validation_loss}")
            self.writer.add_scalars('data/loss', {'train':avg_train_loss, 'valid':avg_validation_loss}, e) # log the losses for tensorboardX
            #Log values and gradients of the parameters (histogram summary)
            for name, param in self.model.named_parameters():
                name = name.replace('.', '/')
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), e)
                self.writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), e)

        self.writer.close()

