# DeepKOALA
**An ultra-fast and accurate tool for KEGG Orthology (KO) assignment, powered by deep learning.**


## About the Project
**DeepKOALA** is a high-performance deep learning-based tool for rapid protein function annotation according to the **KEGG Orthology (KO)** system. By framing KO assignment as an open-set recognition problem, it can effectively distinguish between known and unknown functional sequences, thereby reducing false-positive annotations.

Built on a Gated Recurrent Unit (GRU) architecture, the tool provides excellent computational efficiency while ensuring high accuracy. In this beta version, DeepKOALA offers two operational modes:

* **`full_length` mode**: Delivers high-precision annotation for complete protein sequences.
* **`metagenome` mode**: Specially optimized for handling fragmented sequences common in metagenomic data, significantly improving the recognition rate and accuracy for incomplete sequences.



## Performance

### Comparison with Mainstream Tools

![image](https://github.com/zhaoxi120/deepkoala/blob/main/figures/comparison_with_traditional_tools.png)

On an independent test set , DeepKOALA is up to **37.5 times faster** than BlastKOALA while maintaining a comparable or superior precision (84.13% ) to tools like KofamScan (78.74% ) and GhostKOALA (83.06% ).

### Application on Metagenomic Datasets

![image](https://github.com/zhaoxi120/deepkoala/blob/main/figures/comparison_metagenome.png)

**`metagenome` mode** is optimized for fragmented sequences. It can annotate the 46 million proteins of the OM-RGC v2 catalog in approximately 30 minutes and uniquely identifies over 1 million sequences missed by other mainstream tools.




## Installation

### Prerequisites
- [x] Git
- [x] Python >= 3.9
- [x] (For GPU users) NVIDIA graphics driver


### 1. Clone the Repository

First, clone the source code from GitHub to your local machine and navigate into the project directory.

```bash
git clone https://github.com/zhaoxi120/deepkoala
cd deepkoala
```

### 2. Create and Activate the Virtual Environment

Create an independent Python virtual environment named deepkoala_env inside the project directory.

For MacOS/Linux users:
```bash
python3 -m venv deepkoala_env
source deepkoala_env/bin/activate
```

For Windows users:
```bash
python -m venv deepkoala_env
.\deepkoala_env\Scripts\activate
```

After activation, you will see (deepkoala_env) at the beginning of your terminal prompt.


### 3. Install Dependencies

We use the `requirements.txt` file to manage most of the project's dependencies. Run the following command to install them:
```bash
pip install -r requirements.txt
```

> [!WARNING]
> #### For GPU Users:
> 1. Run `nvidia-smi` in your terminal to find the maximum CUDA Version your driver supports.
> 2. Visit the [Official PyTorch Website](https://pytorch.org/) and select a CUDA platform that is less than or equal to your driver's version limit.
> 3. Open the `requirements.txt` file and replace the existing `torch` line with the compatible version you just found. 
> ```bash
> numpy==1.26.3
> pandas==2.2.2
> torch==2.4.1+cu121 # replace this line
> tqdm==4.66.4
> ```

### 4. Download Pre-trained Models

The pre-trained model file (version February 2025)  is already included in this beta version of the project. No separate download is required.


## Usage
