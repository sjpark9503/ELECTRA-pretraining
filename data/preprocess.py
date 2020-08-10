from xopen import xopen
import os
from tqdm.notebook import tqdm

total_file = ""
files = os.listdir("./openwebtext")

for i in tqdm(range(len(files))): 
    file = files[i]
    with xopen(os.path.join("openwebtext", file)) as f:
        if (".xz" not in file): continue
        print(f"working on file: {file}")
        content = f.read()
        total_file += content
    f.close()

with open("openwebtext.txt", "w") as f:
    f.write(total_file)
f.close()