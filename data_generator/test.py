import numpy as np
# NOTE: run file from parent folder
files = [
    r'./experimental/data/captures/102302.npy',
    r'./experimental/data/captures/dhs-logo.npy'
]

for file in files:
    data = np.load(file)
    name = file.replace("./experimental/data/captures/", "").replace(".npy", "")
    print(name, ": ", data.shape, sep="")
