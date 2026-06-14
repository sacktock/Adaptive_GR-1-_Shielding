import os

if __name__ == "__main__":

    algos = ["PPOLag"]

    for algo in algos:
        template = f"python run_seaquest_masa.py --algo {algo} --tensorboard --runs 2"
        os.system(template)