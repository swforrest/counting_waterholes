import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import os
import utils.testing as val_utils
import argparse
import yaml


def init_run(config_file: str):
    # create a new folder
    os.makedirs(os.path.join("runs", "val"), exist_ok=True)
    # name it "val" + the next number
    run_num = 0
    while os.path.exists(os.path.join("runs", "val", "val" + str(run_num))):
        run_num += 1
    run_folder = os.path.join("runs", "val", "val" + str(run_num))
    os.makedirs(run_folder)
    # make a "imgs" and "plots" folder
    os.makedirs(os.path.join(run_folder, "imgs"))
    os.makedirs(os.path.join(run_folder, "plots"))
    # copy the config file to the folder
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(run_folder, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    # return the folder
    return run_folder


def run_tasks(tasks: dict, config: dict, level=0):
    for task in tasks:
        if tasks[task] == False:
            print(f"{'  ' * level}Task '{task}' disabled, skipping...")
            continue
        elif type(tasks[task]) == dict:
            print(f"{'  ' * level}Task '{task}' has subtasks, running...")
            run_tasks(tasks[task], config, level + 1)
        else:
            try:
                do_task = getattr(
                    val_utils, task
                )  # each task is a funcion in val_utils
                do_task(folder, config)  # Do the task, each takes the folder and config
                # print a green checkmark
                print(f"{'  ' * (level+1)}\033[92m" + "\u2713" + "\033[0m" + f" {task}")
            except AttributeError as e:
                print(
                    f"{'  ' * (level + 1)}\033[91m" + "\u2717" + "\033[0m" + f" {task}"
                )
                # print a red X
                print(f"{'  ' * level}Task '{task}' not found, task must be one of:")
                print(
                    f"{'  ' * level}{''.join([f'{t}, ' for t in dir(val_utils) if not t.startswith('_')])}"
                )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.description = "Cluster classifications and labels from yolo and manual annotations to compare them. File names must be identical between given arguments"
    argparser.add_argument("-c", "--config", help="Configuration Folder", required=True)
    args = argparser.parse_args()
    folder = init_run(args.config)
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    # Basically go through and complete each task
    tasks = config["tasks"]
    run_tasks(tasks, config)
