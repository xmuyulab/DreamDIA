"""
╔═════════════════════════════════════════════════════╗
║                    gpu_settings.py                  ║
╠═════════════════════════════════════════════════════╣
║         Description: Configuration of GPUs          ║
╠═════════════════════════════════════════════════════╣
║                Author: Mingxuan Gao                 ║
║             Contact: mingxuan.gao@utoronto.ca       ║
╚═════════════════════════════════════════════════════╝
"""

import tensorflow as tf

def set_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
