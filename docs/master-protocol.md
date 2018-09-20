
Building the SmartTag models
===

# Initial checks

    source .venv/bin/activate
    neo4j start
    neo4j status
    ls models
    ls runs
    smtag-graph2th --help
    smtag-meta --help
    ls data
    smtag-meta -Z2 -E1
    python -m smtag.datagen.sdgraph2th -l10 -L120 -X1 -y protein,gene -f dummy_test
    tensorboard --logdir runs &
    ps -a | grep tensoboard

# Data generation

## Small molecules

Enrich examples for small molecules. Generate the data with `-5X` sampling of each example of length `-L1200` characters.

    smtag-neo2xml -l1000 -y molecule -f small_molecule
    smtag-convert2th -X5 -L1200 -X5 -f 5X_L1200_small_molecule

# Gene and proteins

    smtag-neo2xml -l1000 -y gene,protein -f gene_protein
    smtag-convert2th -X5 -L1200 -X5 -c gene_protein -f 5X_L1200_gene_protein
    smtag-meta -Z128 -E150 -R0.001 -f 5X_L1200_gene_protein_train

# Cellular components


# Cells


# Tissue & Organs


# Organisms


# Experimental assays


# Diseases

Using a corpus in the brat format (`-b`) instead of the xml format.

    smtag-convert2th -L1200 -X5 -c 

# Intervention and measurements on geneproducts given reporter

Anonymize gene and proteins (-`A`) but do not anonymize reporters (`-AA`). Keep roles only for genes and proteins (`-s`).

    smtag-neo2xml -l1000 -y gene,protein -A gene,protein -AA reporter -s

# Intervention and measurements on small molecules

# Report on geneproducts

Select enrich for protein and genes (`-y`) and reporters (`-r`). Remove all tags except genes and proteins (`-e`).

    smtag-neo2xml -l1000 -y gene,protein -r reporter -e


# B. Moving training data to Amazon EFS

    cd ../cloud
    scp -i basicLinuxAMI.pem 5X_L1200_small_molecule_train.zip ec2-user@smtag-web:/efs/smtag/data/
    scp -i basicLinuxAMI.pem 5X_L1200_gene_protein_train.zip ec2-user@smtag-web:/efs/smtag/data/
    scp -i basicLinuxAMI.pem 5X_L1200_subcell_train.zip ec2-user@smtag-web:/efs/smtag/data/
    scp -i basicLinuxAMI.pem 5X_L1200_cell_train.zip ec2-user@smtag-web:/efs/smtag/data/
    scp -i basicLinuxAMI.pem 5X_L1200_tissue_train.zip ec2-user@smtag-web:/efs/smtag/data/
    scp -i basicLinuxAMI.pem 5X_L1200_organism_train.zip ec2-user@smtag-web:/efs/smtag/data/
    scp -i basicLinuxAMI.pem 5X_L1200_exp_assay_train.zip ec2-user@smtag-web:/efs/smtag/data/
    scp -i basicLinuxAMI.pem 5X_L1200_disease_train.zip ec2-user@smtag-web:/efs/smtag/data/


# C. Training models

# Entity models

### Small molecule model
150 epoch with learning rate 0.001 and batch size 128

    smtag-meta -Z128 -E150 -R0.001 -f 5X_L1200_small_molecule_train

### Gene product model


### Cellular component model

### Cell model

### Tissue and organ model

### Organism model

### Experimental assay model

### Disease model 

## Roles

### Roles of gene products

### Roles of small molecules

### Reporter gene products


```
python -m smtag.train.meta -f 5X_L1200_small_molecule_w_train -E120 -R0.01 -Z128 
```


2. Benchmark on the test set, without tokenization (option `-T`)


```
    python -m smtag.train.evaluate -f 5X_L1200_small_molecule_w_test -m 5X_L1200_small_molecule -T
```

# D. Copy models locally

# E. Evaluate models with test data


