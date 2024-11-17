# Scale-VGM

**Scale-VGM** is a scalable and robust data processing pipeline designed to handle large datasets. It leverages **Bayesian Gaussian Mixture Models (BGMM)** for advanced scaling and normalization, ensuring data quality and consistency. The pipeline uses **Dask** for distributed and parallel computation, making it efficient for processing vast datasets in-memory or across clusters. 

This pipeline is particularly useful for preprocessing data for machine learning models and analytical workflows.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [What the Code Does](#what-the-code-does)
5. [Setup and Installation](#setup-and-installation)
6. [Usage](#usage)
7. [Docker Setup](#docker-setup)
8. [Contributing](#contributing)
9. [License](#license)

---

## Overview

### What is Scale-VGM?

Scale-VGM is a **data transformation pipeline** that focuses on scaling the `Amount` column in datasets using **Bayesian Gaussian Mixture Models (BGMM)**. It:

1. Identifies clusters in the data distribution.
2. Scales the data within each cluster for normalization.
3. Processes datasets in parallel using Dask for high efficiency and scalability.

---

## Features

- **Distributed Processing**: Handles datasets that don't fit into memory by splitting them into smaller partitions and processing them in parallel.
- **Advanced Normalization**: Uses BGMM to normalize data based on learned clusters, preserving meaningful patterns in the data.
- **Robust Design**: Effectively handles extreme values, outliers, and uneven distributions.
- **Integration Ready**: Modular design for seamless integration into existing ML or data engineering workflows.
- **Containerization**: Fully compatible with Docker for consistent execution across environments.

---

## Project Structure

```plaintext
Scale-VGM/
├── data/                         # Directory for input/output data
│   ├── scaled_data_1b/           # Input data in Parquet format
│   └── transformed_data_1b/      # Output data after transformation
├── notebooks/                    # Jupyter notebooks for experimentation
├── src/                          # Source code directory
│   ├── __init__.py               # Initializes src package
│   ├── data_processor.py         # Dask-based processing of large datasets
│   ├── distributed_data_scaling.py # Script for initial data scaling
│   ├── scale_vgm.py              # Implements BGMM-based scaling
│   └── stats.py                  # Statistical utilities
├── tests/                        # Unit tests
├── Dockerfile                    # Docker container setup
├── main.py                       # Orchestrates the full pipeline
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## What the Code Does

### `src/scale_vgm.py`
- Implements the **`ScaleVGM` class**, which performs BGMM-based scaling:
  - **`fit()`**: Fits the BGMM model to a sample of the data (`Amount` column).
  - **`transform()`**: Normalizes the data based on the clusters identified by the BGMM.

### `src/data_processor.py`
- Contains the `DataProcessor` class for distributed data transformation:
  - Partitions the data using Dask.
  - Applies the fitted `ScaleVGM` model to normalize the `Amount` column.
  - Saves the transformed data as Parquet files.

### `src/distributed_data_scaling.py`
- Prepares the data for processing:
  - Loads data using Dask.
  - Ensures compatibility for downstream scaling.

### `main.py`
- Orchestrates the entire pipeline:
  - **Step 1**: Fits the BGMM model on a sample of the data.
  - **Step 2**: Processes the dataset in parallel using `DataProcessor`.

---

## Setup and Installation

### Prerequisites

- **Python 3.10+**
- **Docker** (optional for containerized execution)

### Installation

1. Clone the Repository:
   ```bash
   git clone https://github.com/Midhilesh4890/Scale-VGM.git
   cd Scale-VGM
   ```

2. Set Up a Virtual Environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install Dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Scripts

1. **Run Initial Scaling**:
   ```bash
   python src/distributed_data_scaling.py
   It scales the dataset to 1 billion rows
   ```

2. **Run the Main Pipeline**:
   ```bash
   python main.py 
   It will run the mode-specific normalization for the 1 billion row dataset efficiently
   ```

**Stats Report (Optional)**:

   ```bash
   python src/stats.py
   It will generate stats report like missing values etc;
   ```
---


This project utilizes:

- **Dask** for distributed computation.
- **Scikit-learn** for BGMM implementation.
- **PyArrow** for efficient Parquet file handling.
- **Docker** for containerization.

---

