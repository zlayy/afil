# afil
A novel Adaptive Feature Interaction Learning (AFIL) model for accurate QoS prediction in cloud service recommendation. It dynamically learns correlations between multi-source context features to improve prediction accuracy, especially in sparse data scenarios.

This repository contains the implementation of our proposed AFIL model using the WS-DREAM dataset. The implementation is based on Python 3 and PyTorch.

📁 Project Structure
├── test1.py # Main execution script
├── evaluate.py # Performance evaluation script
├── head1.py # Main model implementation
├── load_data.py # Data loading utilities
├── preprocessing.py # Data preprocessing scripts
├── result.csv # Consolidated results file
├── Ws-Dream/ # WS-DREAM dataset directory
└── result/ # Individual experiment results
└── ...

🛠️ Core Dependencies
- Python 3.x
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

🚀 Quick Start
1. **Clone the repository**

	git clone https://github.com/zlayy/afil
	cd AFIL_main

2. **Install dependencies**

   	pip install torch numpy pandas scikit-learn matplotlib

3. **Run the main pipeline**
   	python test1.py

📈 Results
Experimental results are stored in:
• result.csv: Consolidated results from all experiments
• Individual experiment folders: Detailed results for each experimental setup

📝 Citation
If you use this code in your research, please cite our paper.
