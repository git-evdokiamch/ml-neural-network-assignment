🧠 Fully Connected Neural Network (From Scratch)

Implementation of a fully connected neural network from scratch using NumPy, including manual backpropagation, along with a comparative implementation in PyTorch.

⸻

🚀 Overview

This project was developed as part of a Machine Learning course and focuses on understanding neural networks at both a theoretical and practical level.

The implementation includes:

* Manual derivation and implementation of backpropagation
* Training a neural network without high-level frameworks
* Evaluation on both synthetic and real-world datasets
* Comparison with a PyTorch-based model

⸻

🛠️ Technologies

* Python
* NumPy
* PyTorch
* Matplotlib
* Scikit-learn

⸻

🧩 Features

* Fully connected neural network (1 hidden layer)
* ReLU activation (hidden layer)
* Sigmoid activation (output layer)
* Binary Cross Entropy loss
* Gradient Descent optimization
* Batch training support
* From-scratch implementation (no ML frameworks for core model)

⸻

📊 Experiments

🔹 Synthetic Data (Blobs & Flower)

* Perfect performance on linearly separable data
* Reduced accuracy on harder distributions
* Non-linear decision boundaries learned for complex datasets

🔹 Breast Cancer Dataset

* Mean Accuracy: ~96% (NumPy)
* Mean Accuracy: ~97% (PyTorch)
* Stable performance across multiple runs

🔹 Architecture Exploration

* Tested different hidden layer sizes
* Observed performance improvements up to a saturation point

⸻

⚖️ NumPy vs PyTorch

FeatureNum                        Py Implementation                       PyTorch Implementation

Backprop                                Manual                             Automatic (autograd)

Flexibility                             Medium                                     High

Learning Value                         Very High                                   High

Ease of Use                            Moderate                                    Easy

⸻

📂 Project Structure
*.
*├── fully_connected_nn.py
*├── experiment1.py
*├── experiment2.py
*├── experiment3.py
*├── report.pdf
*└── README.md

⸻

📘 Key Learning Outcomes

* Deep understanding of backpropagation
* Experience building neural networks from scratch
* Practical comparison between low-level and high-level ML tools
* Insight into how architecture affects performance

⸻

📈 Future Improvements

* Add multiple hidden layers (deep networks)
* Implement different optimizers (Adam, RMSprop)
* Extend to multi-class classification
* Add regularization techniques (Dropout, L2)

⸻

👩‍💻 Author

Evdokia Michailou
BSc Informatics & Telematics

⸻

⭐ Notes

This project is intended for educational purposes but demonstrates strong fundamentals in machine learning and neural network implementation.

⸻
