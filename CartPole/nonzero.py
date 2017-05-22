import numpy as np

A = np.array([[1, 0, 1],
              [0, 5, 1],
              [3, 0, 0]])
nonzero = np.nonzero(A)
# Returns a tuple of (nonzero_row_index, nonzero_col_index)
# That is (array([0, 0, 1, 1, 2]), array([0, 2, 1, 2, 0]))

nonzero_row = nonzero[0]
nonzero_col = nonzero[1]

for row, col in zip(nonzero_row, nonzero_col):
    print("A[{}, {}] = {}".format(row, col, A[row, col]))
"""
A[0, 0] = 1
A[0, 2] = 1
A[1, 1] = 5
A[1, 2] = 1
A[2, 0] = 3
"""

# You can even do this
A[nonzero] = -100
print(A)
"""
[[-100    0 -100]
 [   0 -100 -100]
 [-100    0    0]]
 """

print(np.argwhere(A))

print(np.where(A))