# ShiftHandler - README

This is a implementation of the paper [Modeling Shifting Workloads for Learned Database Systems](https://dl.acm.org/doi/abs/10.1145/3639293) (SIGMOD 2024):


```
@article{wu2024modeling,
  title={Modeling Shifting Workloads for Learned Database Systems},
  author={Wu, Peizhi and Ives, Zachary G},
  journal={Proceedings of the ACM on Management of Data},
  volume={2},
  number={1},
  pages={1--27},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```

## Overview
This project focuses on the use of Replay Buffer to handle workload shifts in query-driven learned database systems.


 **Replay Buffer Strategies**: The following replay buffers are used in the project:

   - **All**: Uses all available queries for replay.
   - **Random Sampling (RS)**: Randomly samples queries from the workload for replay.
   - **Latest**: Uses the latest k queries for replay.
   - **ShiftHandler w/ CBP**: Tries to maintain class balances in the replay buffer.
   - **ShiftHandler w/ LWP**: Prioritizes queries based on calculated loss, aiming to emphasize hard examples.

### File Structure

- **`ShiftHandler`**: Directory containing the code for ShiftHandler.
- **`card_exp`**: Directory containing the code for experiment of cardinality estimation.
- **`cost_exp`**: Directory containing the code for experiment of cost estimation.


### How to run the experiment of cardinality estimation
1. **Please enter [this folder](card_exp).**

2. **Downloads all files in https://github.com/andreaskipf/learnedcardinalities/tree/master/data, and place them in the ./data directory.**

3. **Run the following command:**
   ```bash
    python run_card_exp.py --buffersize 300 --numtrain 1000 --numtest 100 
   ```
4. **Important configuration options:**
   - `--buffersize`: Size of the replay buffer.
   - `--batch`: Batch size for each training step.
   - `--queries`: Total number of queries to use for training.
   - `--numtrain`: Number of training queries per template/task.
   - `--numtest`: Number of test queries per template/task.
   - `--imbalance`: Include this flag if there is class imbalance.
   

4. **Analyzing Results**
 - To ensure a fair comparison, the experiment will be conducted using 10 different random seeds.
 - The overall results will be saved in a .txt file (e.g., card_result_imb_False.txt) in JSON format for each replay buffer approach.

### How to run the experiment of cost estimation
1. **Please enter [this folder](cost_exp).**
2. **You will need to download the query plan file (plans.txt) from [this link](https://drive.google.com/file/d/1wUJlJMtDT14AOXUM4YkgJl2Hvpdm65_i/view?usp=sharing), and place it into the folder.**
3. **Run the following command:**
   ```bash
    python run_cost_exp.py --buffersize 50
   ```
4. **Important configuration options:**
   - `--buffersize`: Size of the replay buffer.
   - `--batch`: Batch size for each training step.
   - `--imbalance`: Include this flag if there is class imbalance.
   

4. **Analyzing Results**
 - To ensure a fair comparison, the experiment will also be conducted using 10 different random seeds.
 - The overall results will be saved in a .txt file (e.g., cost_result_imb_False.txt) in JSON format for each replay buffer approach.

## Contact
If you have any questions, feel free to contact me through email (pagewu@seas.upenn.edu).
