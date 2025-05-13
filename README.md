# FedSCOPE: A Comprehensive Evaluation Framework for Federated Learning in Social Computing

FedSCOPE is a federated learning framework designed to evaluate, implement, and optimize fairness, privacy, robustness, and personalized trust in distributed machine learning. By addressing the human-centric performance requirements of social computing, FedSCOPE provides modular implementations of key federated learning algorithms and supports extensive experimentation across representative scenarios.

---

## 📂 Project Structure

The project is organized into the following main directories and files:

### 🗂️ Main Directories

| Directory            | Description                                                                         |
|-----------------------|-------------------------------------------------------------------------------------|
| `adult/`             | Contains data and resources related to the "Adult" dataset for income prediction.   |
| `adult_code/`        | Code for processing and analyzing the "Adult" dataset.                             |
| `compas/`            | Contains data and resources related to the "COMPAS" dataset for recidivism prediction.|
| `compas_code/`       | Code for processing and analyzing the "COMPAS" dataset.                            |
| `depression/`        | Contains data and resources related to student depression diagnosis datasets.       |
| `depression_code/`   | Code for processing and analyzing depression-related datasets.                     |

---

### 📄 Files in Each `*_code/` Directory

Each `*_code/` directory (e.g., `adult_code/`, `compas_code/`) contains the following files:

- **`data.py`**  
  Script for loading, preprocessing, or generating data for federated learning.

- **`FairFed.py`**  
  Implementation of fairness-aware federated learning to mitigate disparities across sensitive groups.

- **`FedAvg.py`**  
  Classical Federated Averaging algorithm for distributed learning.

- **`FLTrust.py`**  
  Trust-based federated learning strategy, incorporating reliability via server-collected clean datasets.

- **`function.py`**  
  Utility and helper functions for evaluation and optimization.

- **`HYPERPARAMETERS.py`**  
  Configuration file defining hyperparameters for algorithms, datasets, and evaluation metrics.

- **`PriHFL.py`**  
  Privacy-preserving federated learning algorithm featuring server-side evaluation and enhancement.

- **`run.py`**  
  Main script for launching federated learning experiments and evaluations.

---

> **Note**: Not all algorithms are activated by default. Selection and configuration of algorithms can be done in `HYPERPARAMETERS.py`.

---

## 🚀 Getting Started

### 1. Clone the Repository

To start using FedSCOPE, clone the repository:

```bash
git clone https://github.com/YananZHOU5555/FedSCOPE.git
cd FedSCOPE
```

### 2. Install Required Python Packages

FedSCOPE requires Python 3.10.4. Ensure you have it installed on your system. Then, install the required dependencies using:

```bash
pip install -r requirements.txt
```

Below is the list of required Python packages and their versions:

| Package           | Version   |
|--------------------|-----------|
| contourpy          | 1.3.1     |
| cycler             | 0.12.1    |
| filelock           | 3.16.1    |
| fonttools          | 4.55.3    |
| fsspec             | 2024.12.0 |
| Jinja2             | 3.1.5     |
| joblib             | 1.4.2     |
| kiwisolver         | 1.4.8     |
| MarkupSafe         | 3.0.2     |
| matplotlib         | 3.10.0    |
| mpmath             | 1.3.0     |
| networkx           | 3.4.2     |
| numpy              | 2.2.1     |
| pandas             | 2.2.3     |
| pillow             | 11.1.0    |
| scikit-learn       | 1.6.0     |
| scipy              | 1.15.0    |
| seaborn            | 0.13.2    |
| torch              | 2.5.1     |

### 3. Run the Project

Navigate to the relevant `*_code` directory and execute the `run.py` script:

```bash
cd adult_code
python run.py
```

You can customize the algorithm or dataset configurations by editing the `HYPERPARAMETERS.py` file.

---

## 📊 Evaluation Components

FedSCOPE introduces a **multi-dimensional evaluation framework** tailored for federated learning in social computing. The evaluation includes:

1. **Learning Performance**: Measures accuracy, recall, and F1-score across diverse data distributions.
2. **Fairness**: Addresses disparities across sensitive attributes using metrics like Equal Opportunity Difference (EOD).
3. **Robustness**: Evaluates resilience to data poisoning and model attacks, quantified by metrics like Drop and Attack Success Rate (ASR).
4. **Reliability**: Measures model stability under non-IID conditions using Relative Performance Deviation (RPD) and Performance Stability Index (PSI).
5. **Privacy Preservation**: Quantifies data exposure risks using the Privacy Preservation Index (PP).
6. **Personalization (Optional)**: Assesses model adaptability to individual participant data, especially for user-facing applications.

---

## 📘 Supported Datasets

FedSCOPE supports the following datasets for experimentation:

1. **Adult**: Predict income levels based on demographic and occupational features.
2. **COMPAS**: Predict the likelihood of recidivism in criminal justice contexts.
3. **Depression**: Diagnose depression risk among students using demographic and behavioral data.

---

Happy experimenting with federated learning! 🚀
```

### Key Changes:
1. Removed all references to `fake_news`, `fake_news_code`, and `flvenv`.
2. Clearly listed the required Python packages in a table under the "Install Required Python Packages" section.
3. Provided installation instructions without the use of a virtual environment.
4. Added clarity around algorithm evaluation, dataset descriptions, and the role of `HYPERPARAMETERS.py`.
