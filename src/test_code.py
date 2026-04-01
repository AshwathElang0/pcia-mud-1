from scipy.spatial.distance import jensenshannon
import numpy as np

# Use the same probability distributions from the previous example
pk = np.array([0.1, 0.4, 0.5])
qk = np.array([0.2, 0.3, 0.5])

# Calculate the Jensen-Shannon distance
js_distance = jensenshannon(pk, qk)
print(f"Jensen-Shannon Distance: {js_distance}")

# The function is symmetric, so jensenshannon(pk, qk) == jensenshannon(qk, pk)
js_distance_rev = jensenshannon(qk, pk)
print(f"Jensen-Shannon Distance (reversed): {js_distance_rev}")
