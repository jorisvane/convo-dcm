import torch

file = open("CLUSTER_TEST.txt", "w")

file.write('Succesfull import torch')
file.write(f'\n{torch.cuda.is_available()}')

file.close