import os

if __name__ == "__main__":

    shield_impls = ["none", "static_1", "static_2", "static_star", "adaptive"]

    for shield_impl in shield_impls:
        template = f"python run_minepump.py --shield {shield_impl} --tensorboard"
        os.system(template)