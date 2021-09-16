#!/usr/bin/env python3
import sys

from torch import tensor
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

import os
from skimage import transform
import torch
import torchvision
from torchvision import transforms#, utils
import numpy as np
import PIL
from PIL import Image
import cv2

import rospkg
rospack = rospkg.RosPack()
sys.path.append(os.path.join(rospack.get_path("u2net_action_pkg"), "u2net"))
import model

MODEL_PATH      = os.path.join(rospack.get_path("u2net_action_pkg"), "u2net/saved_models/u2net")
TEST_IMAGE_PATH = os.path.join(rospack.get_path("u2net_action_pkg"), "u2net/test_data/test_images" )

class U2NET:
    """U2net class
    U2net model 
    """
    def __init__(self, model_pth_name = "u2net.pth", gpu=True) -> None:
        """class initalize function

        Parameters
        ----------
        model_pth_name : str, optional
            saved model path, by default "u2net.pth"
        gpu : bool, optional
            whether to use GPU , by default True
        """
        self.model     = model.U2NET(3,1)
        self.gpu       = gpu
        self.transform = transforms.Compose([RescaleT(320),
                                             ToTensorLab()])


        if gpu==True and torch.cuda.is_available()==False:
            raise Exception("pytorch cannot detect GPU")

        if torch.cuda.is_available() and gpu:
            print("Loading model to GPU...")
            self.model.load_state_dict(torch.load(os.path.join(MODEL_PATH, model_pth_name)))
            self.model.cuda()
        else:
            print("Loading model to CPU...")
            self.model.load_state_dict(torch.load(os.path.join(MODEL_PATH, model_pth_name),map_location='cpu'))
        self.model.eval()
        print("Finish loading")


    def normPRED(self, tensor_mask: torch.Tensor) -> torch.Tensor:
        ma = torch.max(tensor_mask)
        mi = torch.min(tensor_mask)

        tensor_normalized_mask = (tensor_mask-mi)/(ma-mi)

        return tensor_normalized_mask


    def cv2tensor(self, cv_img:np.array) -> torch.Tensor:
        # cv2 -> PIL
        image_pil = PIL.Image.fromarray(cv_img)

        # PIL -> torch.Tensor
        image_tensor = torchvision.transforms.functional.to_tensor(image_pil)
        image_tensor = image_tensor.unsqueeze(dim=0)

        return image_tensor


    def resizeTensorToOriginalShape(self, tensor_img:torch.Tensor, orig_size:tuple) -> np.array:
        pil_img = Image.fromarray(tensor_img*255).convert('RGB')
        pil_img = pil_img.resize((orig_size[1], orig_size[0]),resample=Image.BILINEAR)
        cv_img  = np.array(pil_img, dtype=np.uint8)
        cv_img  = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        return cv_img


    def __call__(self, cv_bgr:np.array) -> np.array:
        orig_size  = cv_bgr.shape[:2]
        cv_lab     = self.transform(cv_bgr)
        tensor_lab = torch.from_numpy(cv_lab).type(torch.FloatTensor).unsqueeze(dim=0)

        if torch.cuda.is_available() and self.gpu:
            tensor_lab = tensor_lab.cuda()
        else:
            tensor_lab = tensor_lab.cpu()

        with torch.no_grad():
            d1,_,_,_,_,_,_= self.model(tensor_lab)

        # normalization
        tensor_mask = d1[:,0,:,:]
        tensor_mask = self.normPRED(tensor_mask)
        tensor_mask = tensor_mask.squeeze()
        numpy_mask  = tensor_mask.cpu().detach().numpy()
        cv_mask     = self.resizeTensorToOriginalShape(numpy_mask, orig_size)
        return cv_mask


class RescaleT(object):
	def __init__(self,output_size) -> None:
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,cv_img:np.array) -> np.array:
		tmp_img = transform.resize(cv_img,(self.output_size,self.output_size),mode='constant')
		return tmp_img


class ToTensorLab(object):
    def __init__(self) -> None:
        pass

    def __call__(self, cv_img:np.array) -> np.array:
        tmp_img = np.zeros((cv_img.shape[0],cv_img.shape[1],3))
        image = cv_img/np.max(cv_img)
        if image.shape[2]==1:
            tmp_img[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmp_img[:,:,1] = (image[:,:,0]-0.485)/0.229
            tmp_img[:,:,2] = (image[:,:,0]-0.485)/0.229
        else:
            tmp_img[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmp_img[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmp_img[:,:,2] = (image[:,:,2]-0.406)/0.225

        tmpImg = tmp_img.transpose((2, 0, 1))

        return tmpImg



if __name__ == "__main__":
    test_model = U2NET()
    cv_test =cv2.imread(os.path.join(TEST_IMAGE_PATH, "0003.jpg"))
    cv_mask = test_model(cv_test)

    cv2.imshow("orig", cv_test)
    cv2.imshow("mask", cv_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("SUCCESS")