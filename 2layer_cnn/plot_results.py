import matplotlib.pyplot as plt

# Default MNIST CNN architecture
probabilities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
errors = [0.0075, 0.0118, 0.0149, 0.016, 0.0246, 0.0291, 0.05, 0.1079, 0.2667, 0.9139, 0.9992]

plt.title("Guan's Noisy Label Experiment")
plt.xlabel("Probability Labels are Made Incorrect")
plt.ylabel("Classification Error Rate")
plt.plot(probabilities, errors)
plt.show()

# Guan's MNIST CNN architecture
probabilities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
errors = [0.0129, 0.0179, 0.0209, 0.0218, 0.0274, 0.0302, 0.0329, 0.0537, 0.0998, 0.9054, 0.9998]

plt.title("Guan's Noisy Label Experiment")
plt.xlabel("Probability Labels are Made Incorrect")
plt.ylabel("Classification Error Rate")
plt.plot(probabilities, errors)
plt.show()

