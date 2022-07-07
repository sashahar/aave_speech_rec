import os
import pandas as pd
import subprocess

MANIFEST_PATH = 'manifests_wsj/temp/train_all.csv'

def create_output_dirs(output_path):
    components = output_path.split("/")[1:-1] #discard file, keep only directories
    curr_path = "/" + components[0]
    #print("full output path: ", output_path)
    if not os.path.exists(curr_path):
        #print(curr_path)
        os.mkdir(curr_path)
    for i in range(1, len(components)):
        curr_path += "/" + components[i]
        if not os.path.exists(curr_path):
            os.mkdir(curr_path)

if __name__ == '__main__':
    df = pd.read_csv(MANIFEST_PATH, names = ['input_filepath', 'output_filepath', 'id'])
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print("written {} files...".format(i))
        input_path = row.input_filepath
        output_path = row.output_filepath
        #print("input path: {}, output path: {}".format(input_path, output_path))
        create_output_dirs(output_path)
        args = ["sph2pipe", "-f", "wav", input_path, output_path]
        subprocess.call(args)
