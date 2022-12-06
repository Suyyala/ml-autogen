from torch.utils.data import Dataset

class GPT2Dataset(Dataset):
    def __init__(self, data, preprocessor):
        self.data = data
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # get the input and target sequences for the sample
        input_seq, target_seq = self.data[index]
        
        # pre-process the input and target sequences
        inputs, targets = self.preprocessor(input_seq, target_seq)
        
        # return the pre-processed input and target sequences
        return inputs, targets
