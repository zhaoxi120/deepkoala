# Resource data notice

## KO definitions in `202502/ko_description.csv`

The file `202502/ko_description.csv` contains KEGG Orthology (KO) definitions from KEGG. These data are copyrighted by Kanehisa Laboratories and are not covered by the DeepKOALA source code's [MIT License](../LICENSE).

Academic and non-academic use of these data is subject to the applicable KEGG licensing terms. Non-academic use requires an appropriate commercial license. Please refer to the [KEGG Copyright and Disclaimer](https://www.kegg.jp/kegg/legal.html) and [KEGG Commercial Licensing](https://www.pathway.jp/en/licensing.html).

This notice applies specifically to the KO definition data. It does not change the licensing terms of other resource files or grant additional rights to use or redistribute KEGG data.

The `202502` directory identifies the associated DeepKOALA model release. It does not establish the retrieval date or database snapshot date of the definitions; those dates are not recorded for this file.

## Using descriptions with model releases

The February 2025 model resources and their description file support `--date 202502 --description`. Without `--description`, inference does not require a description file.

Newer model resources are distributed through [GenomeNet](https://www.genome.jp/ftp/db/deepkoala/). Each release used with `--description` must supply its own `resources/YYYYMM/ko_description.csv`, with `ko` and `description` columns. The program uses the file in the selected model's directory and does not fall back to another release's definitions. Obtain and use each release's data under its applicable provider terms.
