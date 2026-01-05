#Code author: Yunru Zheng
#Date: December 28, 2025

import numpy as np
import matplotlib.pyplot as plt
from src.model import m_func, grad_m_func
from src.optimize import run_sgd_optimization

N_SAMPLES = 50000000
CLIP_THRESHOLD = 10000.0
SIGMA = 1

experiments = [
    # Figure 4: Large Learning Rate, No Batch
    {
        "title": "Figure 4: t=0.004, Batch=1 (High Noise)",
        "lr": 0.004, "batch": 1, "steps": 200, "ylim": (1.0, 4.0)
    },
    # Figure 5: Small Learning Rate, No Batch
    {
        "title": "Figure 5: t=0.001, Batch=1 (Smaller Noise)",
        "lr": 0.001, "batch": 1, "steps": 800, "ylim": (1.0, 4.0)
    },
    # Figure 6: Large Learning Rate, Batch 100
    {
        "title": "Figure 6: t=0.004, Batch=100 (Smoother)",
        "lr": 0.004, "batch": 100, "steps": 100, "ylim": (0.0, 4.0)
    },
    # Figure 7: Small Learning Rate, Batch 1000
    {
        "title": "Figure 7: t=0.001, Batch=1000 (Smoothest)",
        "lr": 0.001, "batch": 1000, "steps": 900, "ylim": (0.0, 4.0)
    },
]


def generate_plots():

    for exp in experiments:
        print(f"Generating {exp['title']}")

        # New plot
        plt.figure(figsize=(8,6))

        # Run 10 independent trajectories
        for trial in range(10):
            history, _ = run_sgd_optimization(
                k=2,
                grad_func = grad_m_func,
                loss_func = m_func,
                batch_size=exp["batch"],
                learning_rate=exp["lr"],
                is_reparam=False,
                num_iterations=exp["steps"],
                start_sigma=1.0)
            plt.plot(history, linewidth=1.5, alpha=0.6)

        plt.title(exp["title"], fontsize=12, fontweight='bold')
        plt.xlabel("Iteration Number")
        plt.ylabel("Sigma")
        plt.ylim(exp["ylim"])
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.axhline(y=2.2, linewidth=1, label='Optimal (Target)')
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

generate_plots()




