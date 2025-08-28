# DeepKOALA
**An ultra-fast and accurate tool for KEGG Orthology (KO) assignment, powered by deep learning.**

## About the Project
**DeepKOALA** is a high-performance deep learning-based tool for rapid protein function annotation according to the **KEGG Orthology (KO)** system. By framing KO assignment as an open-set recognition problem, it can effectively distinguish between known and unknown functional sequences, thereby reducing false-positive annotations.

Built on a Gated Recurrent Unit (GRU) architecture, the tool provides excellent computational efficiency while ensuring high accuracy. In this beta version, DeepKOALA offers two operational modes:

* **`full_length` mode**: Delivers high-precision annotation for complete protein sequences.
* **`metagenome` mode**: Specially optimized for handling fragmented sequences common in metagenomic data, significantly improving the recognition rate and accuracy for incomplete sequences.


## Performance

### Comparison with Mainstream Tools

DeepKOALA strikes an excellent balance between speed and accuracy. On an independent test set of 60 species, it is up to **37.5 times faster** than BlastKOALA and more than 6 times faster than both GhostKOALA and KofamScan. While maintaining this exceptional speed, its precision (84.13%) is comparable or superior to mainstream tools like KofamScan (78.74%) and GhostKOALA (83.06%), achieving an optimal combination of computational efficiency and annotation performance.


### Application on Metagenomic Datasets

DeepKOALA performs exceptionally well when processing complex metagenomic data. The specialized `metagenome` mode not only achieves higher accuracy on fragmented sequences but also recognizes substantially more sequences than the standard model.

The model demonstrates outstanding scalability, **completing the full annotation of the Ocean Microbial Gene Catalog (OM-RGC v2), which contains 46 million proteins, in approximately 30 minutes on a single NVIDIA H100 GPU**. More importantly, DeepKOALA not only identifies a core set of proteins co-annotated by other tools but also **uniquely annotates over 1 million sequences** missed by other methods, highlighting its significant potential for discovering novel functions in complex datasets.

