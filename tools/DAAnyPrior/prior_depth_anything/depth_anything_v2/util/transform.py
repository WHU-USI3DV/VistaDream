import torch
import cv2
import numpy as np

class Resize(object):
    def __init__(
        self, 
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method='bilinear',
    ):
        self.__width = width
        self.__height = height
        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method
        
    def get_size(self, width, height):
        scale_height = self.__height / height
        scale_width = self.__width / width
        
        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise NotImplementedError()
            
        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        else:
            raise NotImplementedError()
        
        return (new_width, new_height)
    
    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(np.int32)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(np.int32)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(np.int32)

        return y

    def __call__(self, sample):
        width, height = self.get_size(sample["image"].shape[-1], sample["image"].shape[-2])
        sample["image"] = torch.nn.functional.interpolate(
            sample["image"], (int(height), int(width)), mode=self.__image_interpolation_method)
        
        return sample
    
class NormalizeImage(object):
    def __init__(self, mean, std, device='cpu'):
        self.__mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.__std = torch.tensor(std).view(1, 3, 1, 1).to(device)

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std
        return sample