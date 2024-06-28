import torch
x = torch.arange(1., 13)

z = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11,]])
print(z)
print(x)

n = 4

c = x.unfold(0, n, n)

print(c)

if x.size(0) % n != 0:
    d = x[x.size(0) - (x.size(0) % n):]
    print(d)
