# Predicting Airline Flight Delays with Apache Spark

This project uses Apache Spark to analyze the 2015 Airline On-Time Performance Dataset. The primary goal is to perform a distributed statistical analysis to identify key factors and to build a Gradient-Boosted Tree (GBT) classification model to predict the likelihood of a significant flight delay.

---

## 1. Project Description

The U.S. airline industry generates a massive volume of operational data daily. Analyzing this data to identify the root causes of flight delays and predict their duration is a computationally intensive task that is often too slow for a single computer. This project aims to address this challenge by using Apache Spark to perform scalable data processing and machine learning.

Our primary goals are:
* To conduct a distributed correlation analysis to identify the key factors—such as carrier, time of day, and flight distance—that are most strongly associated with flight arrival delays.
* To build, train, and evaluate a Gradient-Boosted Tree (GBT) classification model. The model's objective is to predict the likelihood of a significant delay (defined as > 15 minutes), aligning with the project's focus on classification.
* To benchmark the computational performance of this ML task, comparing the speed of a single-node setup versus a two-node cluster.

The project highlights Spark's ability to efficiently manage and analyze large-scale datasets across a cluster, providing timely insights that are critical for operational planning in the aviation industry.

---

## 2. Dataset

This project uses the **Airline On-Time Performance Dataset from 2015**.

* **Source:** [Kaggle: US Flight Delays](https://www.kaggle.com/datasets/usdot/flight-delays)
* **File needed:** `flights.csv`

---

## 3. Environment Setup (Two-VM Cluster)

This project is designed to run on a two-node Apache Spark cluster running Fedora Linux.

* **VM 1 (Master):** `hadoop1` (e.g., `192.168.13.143`)
* **VM 2 (Worker):** `hadoop2` (e.g., `192.168.13.144`)

### On Both VMs (`spark-master` and `spark-worker`)


1.  **Install Python & Required Libraries:**
    The Python environment *must be identical* on all nodes.
    ```bash
    sudo dnf install python3-pip -y
    pip3 install pyspark
    pip3 install numpy 
    ```
    *Note: `numpy` is a required dependency for `pyspark.ml` and must be installed manually.*

2.  **Download & Install Spark:**
    ```bash
    # ON MASTER VM
    # Navigate to your preferred install directory
    cd /opt
    # Download Spark 3.5.1 (or your preferred version)
    sudo wget https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
    # Unpack it
    sudo tar czf spark-3.5.1-bin-hadoop3.tgz
    # Secure Copy it to preferred install directory on Worker VM
    sudo scp spark.tar.gz sat3812@hadoop2:/opt

    # ON WORKER VM
    # Navigate to your preferred install directory
    cd /opt
    # Unpack Spark
    tar xvzf spark.tar.gz
    ```

3.  **Configure Networking:**
    Edit `/etc/hosts` on **both** machines to resolve the hostnames.
    ```bash
    sudo nano /etc/hosts
    ```
    Add the IPs for your VMs:
    ```
    192.168.13.143  spark-master
    192.168.13.144  spark-worker
    ```
    *Note: Change current `hadoop1` and `hadoop2` default IP addresses to the IPs of your two machines.*

4.  **Enable Passwordless SSH (Master-to-Worker):**
    The master node needs to be able to SSH into the worker to start its processes.
    ```bash
    # Run this ON THE MASTER VM ONLY
    ssh-keygen -t rsa
    ssh-copy-id spark-worker
    ```
    Test by running `ssh spark-worker`. You should log in without a password.


### Start the Cluster

1.  **From `spark-master`**, run:
    ```bash
    sudo /opt/spark/sbin/start-all.sh
    ```
2.  **Verify** by opening the Spark Master Web UI in your browser: **`http://spark-master:8080`**. You should see both "ALIVE" workers listed.

---

## 4. How to Run the Analysis

The script `flight_classification.py` is configured for performance comparison.

### Step 1: Place the Data File

Spark executors read data from their *local* filesystem. The `flights.csv` file must exist at the **exact same path** on all nodes in the cluster.

1.  Create a project directory on the **master** node:
    ```bash
    mkdir ~/flight_project
    cd ~/flight_project
    # Move the flight_analysis.py and flights.csv files into this directory
    ```

2.  Copy the data file and directory structure to the **worker** node:
    ```bash
    # From the master node
    ssh spark-worker "mkdir ~/flight_project"
    scp ~/flight_project/flights.csv spark-worker:~/flight_project/
    ```

### Step 2: Run Using Both Machines

1.  **Ensure the script is in cluster mode:**
    Open `flight_classification.py` and confirm the `SparkSession` builder points to your master:
    ```python
    spark = SparkSession.builder \
        .appName("FlightDelayAnalysis") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    ```

2.  **Submit the Job:**
    From your `~/flight_project` directory on `spark-master`, run `spark-submit`.
    ```bash
    spark-submit flight_classification.py
    ```

### Step 3: Run Using One Machine (for Comparison)

1.  **Stop the cluster** (so you are only using one node):
    ```bash
    sudo /opt/spark/sbin/stop-all.sh
    ```

2.  **Start the Master Only**:
    ```bash
    sudo /opt/spark/sbin/start-master.sh
    ```

3.  **Start the Master Only**:
    ```bash
    sudo /opt/spark/sbin/start-worker.sh spark://hadoop1:7077
    ```

4.  **Submit the Job:**
    From your `~/flight_project` directory on `spark-master`, run `spark-submit`.
    ```bash
    spark-submit flight_classification.py
    ```

---

## 5. Project Findings

### Model Performance

* **Single-Node:**
<img width="232" height="61" alt="OneNodeAPR" src="https://github.com/user-attachments/assets/0194d9ce-e14a-4e79-98bc-c843ace420bd" />
<img width="327" height="22" alt="OneNodeAUC" src="https://github.com/user-attachments/assets/71e5b096-5a45-4afc-a1dc-85aa05285f9b" />


* **Two-Node:**
<img width="238" height="58" alt="TwoNodeAPR" src="https://github.com/user-attachments/assets/fcaf3acd-74c4-44cc-b293-29e2359bdb72" />
<img width="328" height="19" alt="TwoNodeAUC" src="https://github.com/user-attachments/assets/6ec2a89b-8319-4fbe-9b02-84eef0dfcc0f" />


### Execution Time Comparison

* **Single-Node Runtime:** 15 minutes (911.66 secconds)
* **Two-Node Runtime:** 9.8 minutes (583.38 seconds)

### Analysis of Results
A performance increase (~43%) was observed. An AUC of ~0.936 shows a high-quality model that can effectively distinguish between delayed and on-time flights. An Accuracy of ~0.935 is an excellent result, and the balanced precision/recall scores show that the model is not biased.

---

## 6. Common Troubleshooting

* **Error: `ModuleNotFoundError: No module named 'numpy'`**
    * **Cause:** The `pyspark.ml` library requires `numpy`, but it isn't installed on the master or worker node.
    * **Fix:** Install the library on **all** nodes: `pip3 install numpy`.

* **Error: `SparkFileNotFoundException: File file:/.../flights.csv does not exist'`**
    * **Cause:** The executor on the worker node tried to read the data file but couldn't find it on its local disk.
    * **Fix:** The data file must exist at the same path on all nodes. Use `scp` to copy the file to the worker (see "How to Run" Step 1).
 
* **Error: `IllegalArgumentException: ...categorical feature 1 has 628 values.`**
    * **Cause:** The default of 32 bins fails, since the ORIGIN_AIRPORT feature has 628 unique values.
    * **Fix:** The maxBins parameter (`GBTClassifier(..., maxBins=650)`) was set to 650.
 
* **Error: `Command exited with code 137`**
    * **Cause:** This is a Linux **Out-Of-Memory (OOM) Killer** error. The executor process (`StringIndexer` is very memory-intensive) used more memory than the VM allocated, so the OS killed it.
    * **Fix:** Allocate more memory for your machine. 8GB memory was allocated to each machine when this project was run.
