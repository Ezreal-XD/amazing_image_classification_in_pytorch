from model import *


def build_model(model_name, num_classes):
    model_name = eval(model_name)
    return model_name(num_classes=num_classes)
