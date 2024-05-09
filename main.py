#implementing the Euler–Maruyama method and using it to solve the Ornstein–Uhlenbeck process stochastic differential equation

import numpy as np;
import matplotlib.pyplot as plt;

class Model:
	"""constants for stochastic model"""
	THETA = 0.5;
	MU = 2.4;
	SIGMA = 0.09;

def mu(y: float, _t: float) -> float:
	"""implementing Ornstein–Uhlenbeck mu"""
	return Model.THETA * (Model.MU - y);

def sigma(_y: float, _t: float) -> float:
	"""implementing Ornstein–Uhlenbeck sigma"""
	return Model.SIGMA;

def dW(delta_t: float) -> float:
	"""sampling a random number at each call"""
	return np.random.normal(loc=0.0, scale=np.sqrt(delta_t));

def run_simulation():
	"""returning result of a full simulation"""
	T_INIT = 0;
	T_END = 10;
	N = 1000;  #compute @ 1000 grid points
	DT = float(T_END - T_INIT) / N;
	TS = np.arange(T_INIT, T_END + DT, DT);
	assert TS.size == N + 1;

	Y_INIT = 0;

	ys = np.zeros(TS.size);
	ys[0] = Y_INIT;
	for i in range(1, TS.size):
		t = T_INIT + (i - 1) * DT;
		y = ys[i - 1];
		ys[i] = y + mu(y, t) * DT + sigma(y, t) * dW(DT);

	return TS, ys;

def plot_simulations(num_sims: int):
	"""plotting several simulations in one graph"""
	for _ in range(num_sims):
		plt.plot(*run_simulation());

	plt.xlabel("t");
	plt.ylabel("I(t)");
	plt.show();

if __name__ == "__main__":
	NUM_SIMS = 1000;
	plot_simulations(NUM_SIMS);
