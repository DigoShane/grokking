import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt

fractions = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
optimizers = ["adamw","adam","sgd"]
seeds = [0,1,2]
results = {}

for optimizer in optimizers:
    optimizer_results = []
    for frac in fractions:
        seed_results = []
        for seed in seeds:
            print(f"\nOptimizer={optimizer} "
                f"Fraction={frac} "
                f"Seed={seed}"
            )

            cmd = f"""python grokking/cli.py --training_fraction {frac} --optimizer {optimizer} --seed {seed}"""
            process = subprocess.run(cmd,shell=True,capture_output=True,text=True)
            output = process.stdout
            
            print("\nSTDOUT:")
            print(process.stdout)
            print("\nSTDERR:")
            print(process.stderr)

            match = re.search(r"RETURNED_BEST_VAL_ACC=([0-9.]+)",output)
            if match:
                best_acc = float(match.group(1))
            else:
                print("FAILED RUN")
                print(process.stderr)
                best_acc = np.nan

            seed_results.append(best_acc)
            print("BEST ACC =", best_acc)

        mean_acc = np.nanmean(seed_results)
        optimizer_results.append(mean_acc)
        print(
            f"\nMEAN ACC "
            f"Optimizer={optimizer} "
            f"Fraction={frac} "
            f"-> {mean_acc}"
        )

    results[optimizer] = optimizer_results

plt.figure(figsize=(8,6))
for optimizer in optimizers:
    plt.plot(fractions,results[optimizer],marker='o',linewidth=2,label=optimizer.upper())
plt.xlabel("Training Fraction")
plt.ylabel("Best Validation Accuracy")
plt.title("Figure 2 Reproduction")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Figure2.png", dpi=300)
plt.show()

print("\nFINAL RESULTS:\n")
for optimizer in optimizers:
    print(optimizer, results[optimizer])
