# DeepKOALA
**An ultra-fast and accurate tool for KEGG Orthology (KO) assignment, powered by deep learning.**

**DeepKoala** is a deep learning–based and high-performance protein annotation tool that rapidly predicts the functions of input protein sequences, classifying them according to the **KEGG Orthology (KO)** system.  

- Built on the **PyTorch** framework with a **GRU-based model architecture**  
- Provides two operation modes:  
  - **full_length mode**: annotates complete protein sequences with higher prediction accuracy  
  - **metagenome mode**: designed for metagenomic data, enabling annotation of a larger number of protein sequences, suitable for more complex datasets  
- Model weights and configuration files are **updated monthly**, ensuring consistency with the latest **KEGG database**  
- Supports two output formats:  
  - **detail mode**: includes probability, threshold, and predicted labels  
  - **simple mode**: only retains high-confidence predictions  
