import os

if __name__ == "__main__":

    shield_impls = ["none", "naive", "static", "repaired", "adaptive-python"]

    for shield_impl in shield_impls:
        template = f"python run_seaquest.py --shield {shield_impl} --tensorboard"
        os.system(template)