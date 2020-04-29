from Utils import *

n_shots = 8192

''' State Preparation'''
# number of obs. times number of features
N = 8*2

# Vector of data X = [x1, x2, ... , xm]
# where m is the number of obs in the
# dataset and xi is a n-dim vector
X = []
# Random generation of data
for i in range(int(N/2)):
    x1 = np.random.randint(3, 100)
    x2 = np.random.randint(3, 100)
    X.append(x1)
    X.append(x2)

# Plot data according to the structure of vector X
for i in range(0,N,2):
    plt.scatter(X[i], X[i+1], alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()

# Normalize vector data
M = 0
for x in X:
    M = M + x ** 2
print(M)
C = 1

desired_vector = [1 / np.sqrt(M * C) * complex(x, 0) for x in X]
len(desired_vector)
for i in range(0,N,2):
    print(i)
    plt.scatter(desired_vector[i], desired_vector[i+1], alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()

'''Quantum circuit to encode data'''

c = ClassicalRegister(4)
# Create a Quantum Circuit
rows = QuantumRegister(3, 'i')
cols = QuantumRegister(1, 'j')
qc = QuantumCircuit(rows, cols, c)

'''From Nicolò thesis'''
# def normalization_AE:
# Amplitude value for each state
# desired_vector = [
#     1 / np.sqrt(M * C) * complex(x1[0], 0),  # 0000
#     1 / np.sqrt(M * C) * complex(x1[1], 0),  # 0001
#     1 / np.sqrt(M * C) * complex(x2[0], 0),  # 0010
#     1 / np.sqrt(M * C) * complex(x2[1], 0),  # 0011
#     1 / np.sqrt(M * C) * complex(x3[0], 0),  # 0100
#     1 / np.sqrt(M * C) * complex(x3[1], 0),  # 0101
#     1 / np.sqrt(M * C) * complex(x4[0], 0),  # 0110
#     1 / np.sqrt(M * C) * complex(x4[1], 0)   # 0111
#     0,                                       # 1000
#     0,                                       # 1001
#     1 / np.sqrt(M * C) * complex(x_in[0], 0),#1010
#     1 / np.sqrt(M * C) * complex(xt_2[0], 0),#1011
#     0,                                       # 1100
#     # 0,                                     # 1101
#     # 1 / np.sqrt(M * C) * complex(x_in[1], 0),# 1110
#     # 1 / np.sqrt(M * C) * complex(xt_2[1], 0),# 1111 ]
# ]


# Initialize Amplitudes into the system
qc.initialize(desired_vector, [rows[0], rows[1], rows[2], cols[0]])
qc.barrier()
print(qc)

#Measurement
qc.measure(rows[0], c[0])
qc.measure(rows[1], c[1])
qc.measure(rows[2], c[2])
qc.measure(cols[0], c[3])

## running quantum circuit
counts = exec_simulator(qc, n_shots = n_shots)

'''Post processing'''
quantum =[]
for key in sorted(counts):
    print("%s: %s" % (key, counts[key]))
    quantum.append(counts[key])

# Computation of dataset considering it after normalisation
# and measuring the qubits
quantum = [v/n_shots for v in quantum]
classical = [v.real**2 for v in desired_vector]

# Compute the difference between the two dataset
# 'classical vs quantum'
R = [np.round(x1 - x2, 4) for (x1, x2) in zip(quantum, classical)]
print(R)
average_error = np.mean([abs(el) for el in R])
## Plot the comparison between the two datasets
for i in range(0,N,2):
    plt.scatter(quantum[i], quantum[i+1], color = 'green', alpha=0.5)
    plt.scatter(classical[i], classical[i + 1], color = 'red', alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()


'''Quantum circuit to encode data'''
c = ClassicalRegister(2)
# Create a Quantum Circuit
rows = QuantumRegister(3, 'i')
cols = QuantumRegister(1, 'j')
qc = QuantumCircuit(rows, cols, c)

# Initialize Amplitudes into the system
qc.initialize(desired_vector, [rows[0], rows[1], rows[2], cols[0]])
qc.barrier()
print(qc)

#Measurement
qc.measure(rows[2], c[0])
qc.measure(cols[0], c[1])

## running quantum circuit
counts = exec_simulator(qc, n_shots = n_shots)

'''Post processing'''
quantum =[]
for key in sorted(counts):
    print("%s: %s" % (key, counts[key]))
    quantum.append(counts[key])

# Computation of dataset considering it after normalisation
# and measuring the qubits
quantum = [v/n_shots for v in quantum]
classical = [v.real**2 for v in desired_vector]
# Compute the difference between the two dataset
# 'classical vs quantum'
R = [np.round(x1 - x2, 4) for (x1, x2) in zip(quantum, classical)]
print(R)
average_error = np.mean([abs(el) for el in R])
## Plot the comparison between the two datasets
for i in range(0,N,2):
    plt.scatter(classical[i], classical[i + 1], color = 'red', alpha=0.5)
for i in range(0,4,2):
    plt.scatter(quantum[i], quantum[i+1], color = 'green', alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()
