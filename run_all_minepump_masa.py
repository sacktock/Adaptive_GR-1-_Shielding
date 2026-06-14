import os

if __name__ == "__main__":

    algos = ["CPO", "PPOLag"]

    for algo in algos:
        template = f"python run_minepump_masa.py --algo {algo} --tensorboard"
        os.system(template)