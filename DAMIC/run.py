import os

# Default parameter
for seed in range(0, 10):
    cmd = 'python main.py --seed {}'.format(seed)
    print(cmd)
    os.system(cmd)


# Parameter analysis K
for k in [5, 15, 20, 25, 30]:
    cmd = 'python main.py --k {} --pretrain True'.format(k)
    print(cmd)
    os.system(cmd)
    for seed in range(0, 10):
        cmd = 'python main.py --seed {} --k {}'.format(seed, k)
        print(cmd)
        os.system(cmd)


# Parameter analysis alpha
for alpha in [0.001, 0.01, 1, 10]:
    cmd = 'python main.py --alpha_value {} --pretrain True'.format(alpha)
    print(cmd)
    os.system(cmd)
    for seed in range(0, 10):
        cmd = 'python main.py --seed {} --alpha_value {}'.format(seed, alpha)
        print(cmd)
        os.system(cmd)












