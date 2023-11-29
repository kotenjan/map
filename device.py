import torch

class Device:
    def __init__(self):
        print("Is CUDA available: ", torch.cuda.is_available())
        print("Number of GPUs available: ", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("GPU Name: ", torch.cuda.get_device_name(0))
        
        # Move models to GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')