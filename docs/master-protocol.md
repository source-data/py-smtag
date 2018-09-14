
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


# A. Data generation

## Small molecules

Generate the data with `-5X` sampling of each example of length `-L1200` characters with a random window method.

    python -m smtag.datagen.sdgraph2th -l1000 -L1200 -X5 -y molecule -f 5X_L1200_small_molecule


## Gene and proteins

```
    python -m smtag.datagen.sdgraph2th -l1000 -L1200 -X5 -y protein,gene -f 5X_L1200_protein_gene 
```

## Cellular components


```
    python -m smtag.datagen.sdgraph2th -l1000 -L1200 -X10 -y subcellular -f 10X_L1200_subcell
```

## Cells


```
    python -m smtag.datagen.sdgraph2th -l1000 -L1200 -X5 -y cell -f 5X_L1200_cell
```

## Tissue & Organs


```bash
python -m smtag.datagen.sdgraph2th -l1000 -L1200 -X5 -y tissue -f 5X_L1200_tissue
```

## Organisms


```bash
python -m smtag.datagen.sdgraph2th -l1000 -L1200 -X5 -y organism -f 5X_L1200_organism
```

## Experimental assays


```bash
python -m smtag.datagen.sdgraph2th -l1000 -L1200 -X5 -y exp_assay -f 5X_L1200_exp_assay
```

## Diseases
Using a corpus in the brat format. Note that we need to use the module `ann2th`.


```bash
python -m smtag.datagen.ann2th -L1200 -X5 -f 5X_L1200_disease compendium/ncbi_disease/train
```

# B. Moving training data to Amazon EFS


```bash
cd ../cloud
```


```bash
scp -i basicLinuxAMI.pem 5X_L1200_small_molecule_train.zip ec2-user@smtag-web:/efs/smtag/data/
```


```bash
scp -i basicLinuxAMI.pem 5X_L1200_gene_protein_train.zip ec2-user@smtag-web:/efs/smtag/data/
```


```bash
scp -i basicLinuxAMI.pem 5X_L1200_subcell_train.zip ec2-user@smtag-web:/efs/smtag/data/
```


```bash
scp -i basicLinuxAMI.pem 5X_L1200_cell_train.zip ec2-user@smtag-web:/efs/smtag/data/
```


```bash
scp -i basicLinuxAMI.pem 5X_L1200_tissue_train.zip ec2-user@smtag-web:/efs/smtag/data/
```


```bash
scp -i basicLinuxAMI.pem 5X_L1200_organism_train.zip ec2-user@smtag-web:/efs/smtag/data/
```


```bash
scp -i basicLinuxAMI.pem 5X_L1200_exp_assay_train.zip ec2-user@smtag-web:/efs/smtag/data/
```


```bash
scp -i basicLinuxAMI.pem 5X_L1200_disease_train.zip ec2-user@smtag-web:/efs/smtag/data/
```

# C. Training models

## Entity models

### Small molecule model
120 epoch with learning rate 0.01 and batch size 128


```bash
python meta -E120 -Z128 -R0.01 -o small_molecule -f 5X_L1200_small_molecule_train
```

### Gene product model


    cd data
    unzip 5X_L1200_protein_gene_train.zip
    cd
    bash: cd: data: No such file or directory
    Archive:  5X_L1200_protein_gene_train.zip
    replace 5X_L1200_protein_gene_train.pyth? [y]es, [n]o, [A]ll, [N]one, [r]ename: 


```bash
python -m smtag.train.meta -f 5X_L1200_protein_gene_train -E120 -Z128 -R 0.01 -o geneprod
```

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


