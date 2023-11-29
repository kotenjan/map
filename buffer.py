import random
import torch


class ImageBuffer:
    def __init__(self, buffer_size=50):
        self.buffer_size = buffer_size
        self.num_images = 0
        self.images = []

    def get_images(self, images):
        to_return = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_images < self.buffer_size:
                self.images.append(image)
                self.num_images += 1
                to_return.append(image)
            else:
                if random.uniform(0, 1) > 0.5:
                    random_id = random.randint(0, self.buffer_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    to_return.append(tmp)
                else:
                    to_return.append(image)
        return torch.cat(to_return, 0)