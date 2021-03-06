# Protocol 26 Feb 2020

# Preparation of context-aware embeddings


Dataset: 

    XPath = ".//abstract"
    embedding_out_channels = 32
    corruption_mode =  'token'
    select_POS = ['CC', 'DT', 'IN', 'PDT', 'PRP', 'PRP$', 'RP', 'TO', 'WP','WP$', 'WRB']
    proba_mask = 1

on corpus/oapmc

Hyperparameters:

    HP = HyperparametersCatStack(
        N_layers = 10,
        in_channels = config.nbits,
        out_channels = config.embedding_out_channels,
        hidden_channels = 512,
        kernel  = 7, # 3 # 7 # 9
        padding = 3, # 1 # 3 # 4
        stride = 1,
        dropout_rate = 0.5,
    )

Training:

    python -m vsearch.train datasets/oapmc-abstracts/ -E40 -Z32 -R0.001 # --> best: 2020-02-24-01-31_last_saved.zip, 0.88 recall 0.94 f1 after 21 epochs

scp -i ~/.ssh/id_rsa.pub lemberge@embo-dgx01:/raid/lemberge/vsearch/models/2020-02-24-01-31_last_saved.zip ./models/
cp models/2020-02-24-01-31_last_saved.zip ../py-smtag/rack/2020-02-24-01-31_last_saved.zip


# Preparation of ready-to-train datasets

## __panel-level__

Unmodified panel-level captions:

    python -m smtag.datagen.convert2th -c 191012 -f 10X_L1200_article_embeddings_32 -X10 -L1200 -E ".//sd-panel"

Anonymized geneproduct, except reporters:

    python -m smtag.datagen.convert2th -c 191012 -f 10X_L1200_anonym_not_reporter_article_embeddings_32 -L1200 -X10 -E ".//sd-panel" -y ".//sd-tag[@type='gene']",".//sd-tag[@type='protein']" -e ".//sd-tag[@type='gene']",".//sd-tag[@type='protein']" -A ".//sd-tag[@role='intervention']",".//sd-tag[@role='assayed']"

Anonymized small molecules:

    python -m smtag.datagen.convert2th -c 191012 -f 10X_L1200_molecule_anonym_article_embeddings_32 -L1200 -X10 -E ".//sd-panel" -e ".//sd-tag[@type='molecule']" -A ".//sd-tag[@role='intervention']",".//sd-tag[@role='assayed']"

Disease brat dataset:

    python -m smtag.datagen.convert2th -c NCBI_disease -b -L1200 -X10 -f 10X_L1200_disease_article_embeddings_32


## __figure-level__

Unmodified figure-level captions:

    python -m smtag.datagen.convert2th -c 191012 -f 10X_L1200_figure_article_embeddings_32 -X10 -L1200 -E ".//fig/caption"

Limited to early emboj with consistent panel labels:

    python -m smtag.datagen.convert2th -c emboj_until_2012 -f 10X_L1200_figure_emboj_2012_article_embeddings_32 -X10 -L1200 -E ".//fig/caption"


# Models

## Multi-entities with exp assays

### Panel level model:

Training:

    python -m smtag.train.meta -f 10X_L1200_article_embeddings_32 -E20 -Z32 -R0.005 -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay --kernel 7 --padding 3 --dropout_rate 0.5 --N_layers 10 --hidden_channels 128

Model: `2020-02-27-12-38_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_019.zip`


Benchmarking:

    python -m smtag.train.evaluator 10X_L1200_article_embeddings_32 2020-02-27-12-38_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_019.zip

__Production model (`--production`): `2020-02-29-13-10_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_019.zip`__

### Figure level model:

Training:

    python -m smtag.train.meta -f 10X_L1200_figure_article_embeddings_32 -E30 -Z32 -R0.005 -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay --kernel 7 --padding 3 --dropout_rate 0.5 --N_layers 10 --hidden_channels 128

Model: `2020-03-10-08-23_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_019.zip`


Benchmarking:

    python -m smtag.train.evaluator 10X_L1200_figure_article_embeddings_32 2020-03-10-08-23_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_019.zip

__Production model (`--production`): `2020-03-10-12-05_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_029.zip`__


### Figure and panel level model:

Training:

    python -m smtag.train.meta -f 10X_L1200_figure_article_embeddings_32,10X_L1200_article_embeddings_32  -E40 -Z32 -R0.005 -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay --kernel 7 --padding 3 --dropout_rate 0.5 --N_layers 10 --hidden_channels 128

Model: `2020-03-10-13-31_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_007.zip`


Benchmarking:

    python -m smtag.train.evaluator 10X_L1200_figure_article_embeddings_32,10X_L1200_article_embeddings_32 2020-03-10-13-31_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_007.zip

__Production model (`--production`): `2020-03-10-16-58_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_012.zip`__


## Roles for gene products:

Training:

    python -m smtag.train.meta -f 10X_L1200_anonym_not_reporter_article_embeddings_32 -E20 -Z32 -R0.001 -o intervention,assayed --kernel 7 --padding 3 --dropout_rate 0.5 --N_layers 10 --hidden_channels 128

Model: `2020-02-28-09-42_intervention_assayed_epoch_017.zip`

Benchmarking:

    python -m smtag.train.evaluator 10X_L1200_anonym_not_reporter_article_embeddings_32 2020-02-28-09-42_intervention_assayed_epoch_017.zip

__Production model (`--production`): `2020-02-29-22-47_intervention_assayed_epoch_019.zip`__


## Role for small molecules

Training:

   python -m smtag.train.meta -f 10X_L1200_molecule_anonym_article_embeddings_32 -E20 -Z32 -R0.001 -o intervention,assayed --kernel 7 --padding 3 --dropout_rate 0.5 --N_layers 10 --hidden_channels 128

Model: `2020-02-28-21-07_intervention_assayed_epoch_019.zip`

Benchmarking:

    python -m smtag.train.evaluator 10X_L1200_molecule_anonym_article_embeddings_32 2020-02-28-21-07_intervention_assayed_epoch_019.zip

__Production model (`--production`): `2020-03-01-01-29_intervention_assayed_epoch_019.zip`__


## Reporter

Training:

    python -m smtag.train.meta -f 10X_L1200_article_embeddings_32 -E20 -Z32 -R0.001 -o reporter --kernel 7 --padding 3 --dropout_rate 0.5 --N_layers 3 --hidden_channels 128

Model: `2020-02-28-22-35_reporter_epoch_010.zip`

Benchmarking:

    python -m smtag.train.evaluator 10X_L1200_article_embeddings_32 2020-02-28-22-35_reporter_epoch_010.zip

__Production model (`--production`): `2020-03-01-08-07_reporter_epoch_004.zip`__


## Diseases

Training:

_Note: using 10X_L1200_figure_emboj_2012_article_embeddings_32 as 'decoy' dataset to incread number of negative example, since brat dataset very small. Decoy should not be too large or too confusing (eg include only few or no positives)._

    python -m smtag.train.meta -f 10X_L1200_disease_article_embeddings_32,10X_L1200_figure_emboj_2012_article_embeddings_32 -E40 -Z32 -R0.001 -o disease --kernel 7 --padding 3 --dropout_rate 0.5 --hidden_channel 128

Note: used default N_layers=3 !

Model: `2020-02-29-09-20_disease_epoch_030.zip`

Benchmarking:

    python -m smtag.train.evaluator 10X_L1200_disease_article_embeddings_32,10X_L1200_figure_emboj_2012_article_embeddings_32 2020-02-29-09-20_disease_epoch_030.zip

__Production model (`--production`): `2020-03-01-08-49_disease_epoch_020.zip`__


## Panel segmentation:

Training: 

    python -m smtag.train.meta -f 10X_L1200_figure_emboj_2012_article_embeddings_32 -E20 -Z32 -R0.001 -o panel_stop --kernel 7 --padding 3 --dropout_rate 0.5 --hidden_channel 128

Note: used default N_layers=3 !

Model: `2020-02-29-09-38_panel_stop_epoch_019.zip`

Benchmarking:

    python -m smtag.train.evaluator 10X_L1200_figure_emboj_2012_article_embeddings_32 2020-02-29-09-38_panel_stop_epoch_019.zip

__Production model (`--production`): `2020-03-01-09-21_panel_stop_epoch_012.zip`__


# Benchmarking of **production** models:

<!-- python -m smtag.train.evaluator 10X_L1200_article_embeddings_32 2020-02-29-13-10_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_019.zip

========================================================                                                                                                        
 Data: /workspace/py-smtag/resources/data4th/10X_L1200_article_embeddings_32                                                                                    
 
 Model: 2020-02-29-13-10_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_019.zip                                                           


 Global stats:

        precision = 0.824442982673645   
        recall = 0.79
        f1 = 0.80

 Feature: 'small_molecule'

        precision = 0.75
        recall = 0.75
        f1 = 0.75

 Feature: 'geneprod'

        precision = 0.86
        recall = 0.88
        f1 = 0.87

 Feature: 'subcellular'

        precision = 0.76
        recall = 0.74
        f1 = 0.75

 Feature: 'cell'

        precision = 0.85
        recall = 0.84
        f1 = 0.85

 Feature: 'tissue'

        precision = 0.76
        recall = 0.65
        f1 = 0.70

 Feature: 'organism'

        precision = 0.79
        recall = 0.78
        f1 = 0.79

 Feature: 'assay'

        precision = 0.82
        recall = 0.67
        f1 = 0.74

 Feature: 'untagged'

        precision = 0.99
        recall = 1.00
        f1 = 0.99 -->

---

    python -m smtag.train.evaluator 10X_L1200_figure_article_embeddings_32,10X_L1200_article_embeddings_32 2020-03-10-16-58_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_012.zip

========================================================

 Data: /workspace/py-smtag/resources/data4th/10X_L1200_figure_article_embeddings_32 + /workspace/py-smtag/resources/data4th/10X_L1200_article_embeddings_32

 Model: 2020-03-10-16-58_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_012.zip

 Global stats:

        precision = 0.8236414790153503
        recall = 0.79
        f1 = 0.80

 Feature: 'small_molecule'

        precision = 0.76
        recall = 0.74
        f1 = 0.75

 Feature: 'geneprod'

        precision = 0.85
        recall = 0.89
        f1 = 0.87

 Feature: 'subcellular'

        precision = 0.78
        recall = 0.71
        f1 = 0.74

 Feature: 'cell'

        precision = 0.86
        recall = 0.83
        f1 = 0.85

 Feature: 'tissue'

        precision = 0.71
        recall = 0.73
        f1 = 0.72

 Feature: 'organism'

        precision = 0.80
        recall = 0.76
        f1 = 0.78

 Feature: 'assay'

        precision = 0.83
        recall = 0.66
        f1 = 0.73

 Feature: 'untagged'

        precision = 0.99
        recall = 0.99
        f1 = 0.99

---

    python -m smtag.train.evaluator 10X_L1200_anonym_not_reporter_article_embeddings_32 2020-02-29-22-47_intervention_assayed_epoch_019.zip

========================================================

 Data: /workspace/py-smtag/resources/data4th/10X_L1200_anonym_not_reporter_article_embeddings_32

 Model: 2020-02-29-22-47_intervention_assayed_epoch_019.zip

 Global stats: 

        precision = 0.908376157283783
        recall = 0.91
        f1 = 0.91

 Feature: 'intervention'

        precision = 0.85
        recall = 0.82
        f1 = 0.84

 Feature: 'assayed'

        precision = 0.87
        recall = 0.89
        f1 = 0.88

 Feature: 'untagged'

        precision = 1.00
        recall = 1.00
        f1 = 1.00

---

    python -m smtag.train.evaluator 10X_L1200_molecule_anonym_article_embeddings_32 2020-03-01-01-29_intervention_assayed_epoch_019.zip

========================================================

 Data: /workspace/py-smtag/resources/data4th/10X_L1200_molecule_anonym_article_embeddings_32

 Model: 2020-03-01-01-29_intervention_assayed_epoch_019.zip

 Global stats: 

        precision = 0.9225161075592041
        recall = 0.88
        f1 = 0.90

 Feature: 'intervention'

        precision = 0.93
        recall = 0.97
        f1 = 0.95

 Feature: 'assayed'

        precision = 0.84
        recall = 0.66
        f1 = 0.74

 Feature: 'untagged'

        precision = 1.00
        recall = 1.00
        f1 = 1.00

---

    python -m smtag.train.evaluator 10X_L1200_article_embeddings_32 2020-03-01-08-07_reporter_epoch_004.zip

========================================================

 Data: /workspace/py-smtag/resources/data4th/10X_L1200_article_embeddings_32

 Model: 2020-03-01-08-07_reporter_epoch_004.zip

 Global stats: 

        precision = 0.9482418298721313
        recall = 0.89
        f1 = 0.92

 Feature: 'reporter'

        precision = 0.90
        recall = 0.78
        f1 = 0.83

 Feature: 'untagged'

        precision = 1.00
        recall = 1.00
        f1 = 1.00

---

    python -m smtag.train.evaluator 10X_L1200_disease_article_embeddings_32,10X_L1200_figure_emboj_2012_article_embeddings_32 2020-03-01-08-49_disease_epoch_020.zip

========================================================

 Data: /workspace/py-smtag/resources/data4th/10X_L1200_disease_article_embeddings_32 + /workspace/py-smtag/resources/data4th/10X_L1200_figure_emboj_2012_article_embeddings_32

 Model: 2020-03-01-08-49_disease_epoch_020.zip

 Global stats: 

        precision = 0.9508796334266663
        recall = 0.90
        f1 = 0.93

 Feature: 'disease'

        precision = 0.91
        recall = 0.81
        f1 = 0.86

 Feature: 'untagged'

        precision = 0.99
        recall = 1.00
        f1 = 0.99

---

    python -m smtag.train.evaluator 10X_L1200_figure_emboj_2012_article_embeddings_32 2020-03-01-09-21_panel_stop_epoch_012.zip

========================================================

 Data: /workspace/py-smtag/resources/data4th/10X_L1200_figure_emboj_2012_article_embeddings_32

 Model: 2020-03-01-09-21_panel_stop_epoch_012.zip

 Global stats: 

        precision = 0.9975236654281616
        recall = 0.99
        f1 = 1.00

 Feature: 'panel_stop'

        precision = 1.00
        recall = 0.99
        f1 = 0.99

 Feature: 'untagged'

        precision = 1.00
        recall = 1.00
        f1 = 1.00


Transfer of production models to local rack

       <!-- scp -i ~/.ssh/id_rsa.pub lemberge@embo-dgx01:/raid/lemberge/py-smtag/resources/models/2020-02-29-13-10_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_019.zip . -->

       scp -i ~/.ssh/id_rsa.pub lemberge@embo-dgx01:/raid/lemberge/py-smtag/resources/models/2020-03-10-16-58_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_012.zip .

       scp -i ~/.ssh/id_rsa.pub lemberge@embo-dgx01:/raid/lemberge/py-smtag/resources/models/2020-02-29-22-47_intervention_assayed_epoch_019.zip .

       scp -i ~/.ssh/id_rsa.pub lemberge@embo-dgx01:/raid/lemberge/py-smtag/resources/models/2020-03-01-01-29_intervention_assayed_epoch_019.zip .

       scp -i ~/.ssh/id_rsa.pub lemberge@embo-dgx01:/raid/lemberge/py-smtag/resources/models/2020-03-01-08-07_reporter_epoch_004.zip .

       scp -i ~/.ssh/id_rsa.pub lemberge@embo-dgx01:/raid/lemberge/py-smtag/resources/models/2020-03-01-08-49_disease_epoch_020.zip .

       scp -i ~/.ssh/id_rsa.pub lemberge@embo-dgx01:/raid/lemberge/py-smtag/resources/models/2020-03-01-09-21_panel_stop_epoch_012.zip .
