from torch.utils.data import Dataset, DataLoader
import torch 
import glob
import os 
import numpy as np
from tqdm import tqdm
from datasets.relational.generator import generate

class RandomRelationalDataset(Dataset):
    def __init__(self, mode, combinations, transform) -> None:
        super().__init__()

        self.mode = mode 
        self.combinations = combinations
        self.transform = transform

    def __len__(self):
        return len(self.combinations)
    
    def __getitem__(self, index):
        combi = self.combinations[index]
        image1 = generate(combi[0], combi[1], combi[2])
        image2 = generate(combi[0], combi[1], combi[2])

        image1 = self.transform(image1)
        image2 = self.transform(image2)

        return '-'.join(combi), image1, image2
    
class RepeatEvalRandomRelationalDataset(Dataset):
    def __init__(self, mode, combinations, transform, repeat) -> None:
        super().__init__()
        
        self.mode = mode
        self.combinations = combinations
        self.transform = transform

        self.speaker_img = []
        self.listener_img = []

        for r in tqdm(range(repeat)):
            for combi in combinations:
                image1 = generate(combi[0], combi[1], combi[2])
                image2 = generate(combi[0], combi[1], combi[2])

                image1 = self.transform(image1)
                image2 = self.transform(image2)

                self.speaker_img.append(image1)
                self.listener_img.append(image2)

        self.speaker_img = np.array(self.speaker_img)
        self.listener_img = np.array(self.listener_img)

    def __len__(self):
        return len(self.speaker_img)
    

    def __getitem__(self, index):
        combi = self.combinations[index%len(self.combinations)]
        image1 = self.speaker_img[index]
        image2 = self.listener_img[index]

        return '-'.join(combi), image1, image2
    
