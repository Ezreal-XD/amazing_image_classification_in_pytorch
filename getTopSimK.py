import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torchvision import transforms
from PIL import Image
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict
from utils.metric.metric import get_iou
from utils.convert_state import convert_state_dict


model = build_model("shapenet15_bn_gap", num_classes=3755)

model = model.cuda()  # using GPU for inference

checkpoint = torch.load("checkpoint/hansim/shapenet15_bn_gapbs256gpu1_trainval/model_191.pth")
model.load_state_dict(checkpoint['model'])

model.eval()
cudnn.benchmark = True


img_path = "data/HanSim/test/00000/0.png"

image = Image.open(img_path).convert('RGB')

normMean = [0.25764307, 0.25764307, 0.25764307]
normStd = [0.29107955, 0.29107955, 0.29107955]
normTransform = transforms.Normalize(normMean, normStd)

testTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

image = testTransform(image).cuda()


output = model(image)

print(output.shape)
print(output)

# torch.cuda.synchronize()
#
# output = output.cpu().data[0].numpy()
# output = output.transpose(1, 2, 0)
# output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
