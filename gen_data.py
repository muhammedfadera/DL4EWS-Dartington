#%%
import numpy as np
from matplotlib import pyplot as plt
import random
from utilities import trend
#%%
np.random.seed(16022026)
N = 1000 # number of samples 
seq_length = 5000

# parameter starts at -2 and ends at 0
r_end = 0
r_start = -2
dt = 0.01
sq_dt = np.sqrt(dt) # to be used in Euler-Maruyama time-stepping
T = seq_length*dt
ts = np.linspace(0, T, seq_length)
r = np.linspace(r_start, r_end, seq_length)
# noise standard devation
noise_sds = 0.01

#%% #* Saddlenode data generation
def saddlenode(x, r):
    return r + x**2

X_data_sn = np.zeros((N, seq_length))
trend_sn = -np.sqrt(-r).reshape(1, -1)

x0 = -np.sqrt(-r_start)
X_data_sn[:, 0] = x0
dW = sq_dt*np.random.randn(N, seq_length-1)
for t in range(1, seq_length):
    x = X_data_sn[:, t-1]
    X_data_sn[:, t] = x + dt*saddlenode(x, r[t]) + noise_sds*dW[:, t-1]
#%%
X_detrended_sn = X_data_sn - trend_sn
#%% #* Null data generation

factor = 20 
def null_run(x, r):
    return -x + r/factor
X_data_null = np.zeros((N, seq_length))
trend_null = r.reshape(1, -1) /factor

x0 = r_start/factor 
X_data_null[:, 0] = x0
dW = sq_dt*np.random.randn(N, seq_length-1)

for t in range(1, seq_length):
    x = X_data_null[:, t-1]
    X_data_null[:, t] = x + dt*null_run(x, r[t]) + noise_sds*dW[:, t-1]
#%%
X_detrended_null = X_data_null - trend_null
#%%
train_seq_length = 1500
# %% visualise some examples
idx = random.randint(0, N)
x_sn = X_data_sn[0]
x_sn_cond = x_sn < 2
x_detrended_sn = X_detrended_sn[0][x_sn_cond]
#%%
before_tip = 500
fig, ax = plt.subplots(1, 2, figsize = (10, 4))
ax[0].plot(ts, x_sn[x_sn_cond], label = "SN")
ax[0].plot(ts, trend_sn[0][x_sn_cond], label = "SN trend")
ax[0].plot(ts, X_data_null[0], label = "null")
ax[0].plot(ts, trend_null[0], label = "null trend")
ax[0].set_xlabel("time [s]", fontsize = 12)
ax[0].legend()
ax[1].plot(ts, x_detrended_sn, label = "SN fluctuations")
ax[1].plot(ts, X_detrended_null[0], label = "null fluctuations", color = "green")

ax[1].axvline(ts[seq_length - before_tip], label = "training seq. ends", color = "red")
ax[1].set_xlabel("time [s]", fontsize = 12)
ax[1].legend()
plt.show()
# %%
train_test_idx = np.linspace(0, seq_length-before_tip, train_seq_length, dtype=int)
#%% split into train and test for each type
X_data_sn_train_test = X_data_sn[:, train_test_idx]
X_data_null_train_test = X_data_null[:, train_test_idx]

train_ratio = 0.8
n_train = int(X_data_sn.shape[0] * train_ratio)

X_train_sn = X_data_sn_train_test[:n_train]
X_test_sn = X_data_sn_train_test[n_train:]


X_train_null = X_data_null_train_test[:n_train]
X_test_null = X_data_null_train_test[n_train:]
#%%
np.savetxt("./data/X_train_saddlenode.txt", X_train_sn)
np.savetxt("./data/X_test_saddlenode.txt", X_test_sn)

np.savetxt("./data/X_train_null.txt", X_train_null)
np.savetxt("./data/X_test_null.txt", X_test_null)

# %% #* transcritical data generation
def transcritical(x, r):
    return x**2 - r*x

X_data_tc = np.zeros((N, seq_length))

x0 = r_start
X_data_tc[:, 0] = x0
dW = sq_dt*np.random.randn(N, seq_length-1)
for t in range(1, seq_length):
    x = X_data_tc[:, t-1]
    X_data_tc[:, t] = x + dt*transcritical(x, r[t]) + noise_sds*dW[:, t-1]

#%%
X_data_tc_train_test = X_data_tc[:, train_test_idx]
trend_tc_lowess = np.apply_along_axis(trend, 1, X_data_tc_train_test, method = "Lowess")
trend_sn_lowess = np.apply_along_axis(trend, 1, X_data_sn_train_test, method = "Lowess")
trend_null_lowess = np.apply_along_axis(trend, 1, X_data_null_train_test, method = "Lowess")

X_detrended_tc = X_data_tc_train_test - trend_tc_lowess
X_detrended_sn = X_data_sn_train_test - trend_sn_lowess
X_detrended_null = X_data_null_train_test - trend_null_lowess

X_train_tc = X_detrended_tc[:n_train]
X_test_tc = X_detrended_tc[n_train:]

X_train_sn = X_detrended_sn[:n_train]
X_test_sn = X_detrended_sn[n_train:]

X_train_null = X_detrended_null[:n_train]
X_test_null = X_detrended_null[n_train:]

np.savetxt("./data/X_train_saddlenode_detrended_lowess.txt", X_train_sn)
np.savetxt("./data/X_test_saddlenode_detrended_lowess.txt", X_test_sn)

np.savetxt("./data/X_train_null_detrended_lowess.txt", X_train_null)
np.savetxt("./data/X_test_null_detrended_lowess.txt", X_test_null)

np.savetxt("./data/X_train_transcritical_detrended_lowess.txt", X_train_tc)
np.savetxt("./data/X_test_transcritical_detrended_lowess.txt", X_test_tc)

# %%
trend_tc_gaussian = np.apply_along_axis(trend, 1, X_data_tc_train_test, method = "Gaussian")
trend_null_gaussian = np.apply_along_axis(trend, 1, X_data_null_train_test, method = "Gaussian")
trend_sn_gaussian = np.apply_along_axis(trend, 1, X_data_sn_train_test, method = "Gaussian")

X_detrended_tc = X_data_tc_train_test - trend_tc_gaussian
X_detrended_sn = X_data_sn_train_test - trend_sn_gaussian
X_detrended_null = X_data_null_train_test - trend_null_gaussian

X_train_tc = X_detrended_tc[:n_train]
X_test_tc = X_detrended_tc[n_train:]

X_train_sn = X_detrended_sn[:n_train]
X_test_sn = X_detrended_sn[n_train:]

X_train_null = X_detrended_sn[:n_train]
X_test_sn = X_detrended_sn[n_train:]

np.savetxt("./data/X_train_saddlenode_detrended_gaussian.txt", X_train_sn)
np.savetxt("./data/X_test_saddlenode_detrended_gaussian.txt", X_test_sn)

np.savetxt("./data/X_train_null_detrended_gaussian.txt", X_train_null)
np.savetxt("./data/X_test_null_detrended_gaussian.txt", X_test_null)

np.savetxt("./data/X_train_transcritical_detrended_gaussian.txt", X_train_tc)
np.savetxt("./data/X_test_transcritical_detrended_gaussian.txt", X_test_tc)

