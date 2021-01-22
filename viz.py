#
#Author: zephyr
#Date: 2021-01-04 12:02:45
#LastEditors: zephyr
#LastEditTime: 2021-01-04 12:04:55
#FilePath: \MovieRatingPrediction\viz.py
#
import torch
from DeepModel import RecModel
from torchviz import make_dot

models = RecModel(1024,1024)
net_plot = make_dot(models(x),params = dict(models.named_parameters()))
net_plot.view()