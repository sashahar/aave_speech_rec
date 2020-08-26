import os
import pandas as pd
import subprocess

MANIFEST_PATH = 'manifests_wsj/temp/val.csv'

def create_output_dirs(output_path):
    components = output_path.split("/")[:-1] #discard file, keep only directories
    print(components)
    curr_path = components[0]
    if not os.path.exists(curr_path):
        os.mkdir(curr_path)
    for i in range(1, len(components)):
        curr_path += "/" + components[i]
        if not os.path.exists(curr_path):
            os.mkdir(curr_path)

if __name__ == '__main__':
    df = pd.read_csv(MANIFEST_PATH, names = ['input_filepath', 'output_filepath', 'id'])
    print(df.head())
    for i, row in df.iterrows():
        input_path = row.input_filepath
        output_path = row.output_filepath
        print("input path: {}, output path: {}".format(input_path, output_path))
        create_output_dirs(output_path)
        #subprocess.call(['ls', '-l', '-a'])
        args = ["sph2pipe", "-f", "wav", input_path, output_path]
        print(args)
        subprocess.call(args)
