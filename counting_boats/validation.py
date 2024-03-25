import os
import utils.validation as val_utils
import argparse
import yaml

def init_run(config_file:str):
    # create a new folder
    os.makedirs(os.path.join("runs", "val"), exist_ok=True)
    # name it "val" + the next number
    run_num = 0
    while os.path.exists(os.path.join("runs", "val", "val" + str(run_num))):
        run_num += 1
    os.makedirs(os.path.join("runs", "val", "val" + str(run_num)))
    # make a "imgs" and "plots" folder
    os.makedirs(os.path.join("runs", "val", "val" + str(run_num), "imgs"))
    os.makedirs(os.path.join("runs", "val", "val" + str(run_num), "plots"))
    # copy the config file to the folder
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join("runs", "val", "val" + str(run_num), "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    # return the folder
    return os.path.join("runs", "val", "val" + str(run_num))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.description = "Cluster classifications and labels from yolo and manual annotations to compare them. File names must be identical between given arguments"
    argparser.add_argument("-c", "--config", help="Configuration Folder", required=True)
    args = argparser.parse_args()
    folder = init_run(args.config)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    # Basically go through and complete each task
    tasks = config["tasks"]
    for task in tasks:
        if config["tasks"][task] == False:
            print(f"Task '{task}' disabled, skipping...")
            continue
        do_task = getattr(val_utils, task)  # each task is a funcion in val_utils
        do_task(folder, config)             # Do the task, each takes the folder and config