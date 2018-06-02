from loader import Dataset
from math import floor
import logging
logger = logging.getLogger(__name__)

class Minibatches: #Minibatches(Dataset)?
    """
    Chunks a Dataset of already randomized examples into an array of minibatches.
    Minibatches is iterable and yields successively one minibatch (Dataset).
    Constructor takes:
        minibatch_size (int): the number of examples per minibatch
        dataset (Dataset): the dataset to randomly split into minibatches
    Usage:
        minibatches = Minibatches(dataset, 128)
        for minibatch in minibatches:
            input = m.input
            target = m.output
            prediction = model(input)
            loss = loss_fn(prediction, target)
    """
    def __init__(self, dataset, minibatch_size):

        self.L = dataset.L
        self.nf_input = dataset.nf_input
        self.nf_output = dataset.nf_output
        self.minibatch_size = minibatch_size
        self.minibatch_number = floor(dataset.N / self.minibatch_size) #the rest of the examples will be ignored
        self.minibatches = [] #should we make this private?

        for i in range(self.minibatch_number):
            this_minibatch = Dataset(self.minibatch_size, self.nf_input, self.nf_output, self.L)
            #minibatch_size is 1 for online training 
            start = i * self.minibatch_size
            stop = start + self.minibatch_size
            this_minibatch.input = dataset.input[start:stop, : , : ]
            this_minibatch.output = dataset.output[start:stop, : , : ]
            this_minibatch.text = dataset.text[start:stop]
            this_minibatch.provenance = dataset.provenance[start:stop]
            self.minibatches.append(this_minibatch)
        self.minibatches = iter(self.minibatches) 
            #if CUDA_ON:
                #print("CUDA ON: minibatches input and output tensors as cuda")
                #make them CUDA

    #make it iterable and shuffable
    def __iter__(self):
        return self.minibatches.__iter__()

    def __next__(self):
        return self.minibatches.__next__()
        
    def __len__(self):
        return len(self.minibatches)
        
    def __getitem__(self, i):
        return self.minibatches[i]
        
    def __setitem__(self, i, val):
        self.minibatches[i] = val
    
def tester():
    logger.debug("> testing")
    d = Dataset(4, 32, 10)
    d.text = ["1234567890", "0987654321", "1234567890", "0987654321"]
    d.provenance = ["a", "b", "c", "d"]
    minibatches = Minibatches(d, 2)
    for m in minibatches:
        print(m.text, m.provenance, m.input.size(), m.output.size())
    logger.debug("> All OK")

if __name__ == '__main__':           # Only when run
    tester()                         # Not when imported

