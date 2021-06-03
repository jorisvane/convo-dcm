import torch

file = open("CLUSTER_TEST.txt", "w")

file.write('Succesfull import torch')
file.write(torch.cuda.is_available())

file.close