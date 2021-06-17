import torch
print(torch.__version__)
file = open("CLUSTER_TEST.txt", "w")

file.write('Succesfull import torch')
file.write(f'\n{torch.cuda.is_available()}')

file.close()

print('DONE')