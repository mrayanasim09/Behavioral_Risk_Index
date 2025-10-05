# montecarlo_fast.py
import time
import numpy as np

print("=== Monte Carlo Performance Test ===")
n_simulations = 200_000
print(f"Running {n_simulations:,} Monte Carlo simulations...")

start_time = time.time()

# Vectorized simulation: all 200k × 30 in one go
np.random.seed(42)
simulations = np.random.normal(50, 15, (n_simulations, 30))
simulations = np.clip(simulations, 0, 100)

# Stats
mean_forecast = np.mean(simulations, axis=0)
std_forecast = np.std(simulations, axis=0)
var_95 = np.percentile(simulations[:, 0], 5)
cvar_95 = np.mean(simulations[simulations[:, 0] <= var_95, 0])

duration = time.time() - start_time
print(f"✅ Monte Carlo completed in {duration:.2f} s")
print(f"   - Simulations: {n_simulations:,}")
print(f"   - VaR 95%: {var_95:.2f}")
print(f"   - CVaR 95%: {cvar_95:.2f}")
print(f"   - Speed: {n_simulations/duration:,.0f} sim/s")
