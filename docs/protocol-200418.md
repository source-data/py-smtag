# Models based on UNET (19 April 2020)

# Embedding model

Hyperparameters:

    model_class = Unet1d
    HP = HyperparametersUnet(
        in_channels = config.nbits,
        out_channels = config.embedding_out_channels,
        nf_table = [64, 128, 256, 512, 512],
        kernel_table  = [3, 3, 3, 3],
        stride_table = [1, 1, 1, 1],
        dropout_rate = 0.5,
        pool = True
    )

Training:

    python -m vsearch.train datasets/oapmc_abstracts/ -E40 -Z32 -R0.001 --nn unet # --> recall=0.88, f1=0.93 saved as: 2020-04-18-23-03_final.zip


# Preparation of ready-to-train datasets

## __panel-level__

Unmodified panel-level captions:

    python -m smtag.datagen.convert2th -c 191012 -f 10X_L1200_article_embeddings_unet_32 -X10 -L1200 -E ".//sd-panel"


Unmodified panel-level captions with noise:

    _corrupt_proba     = 0.01

    python -m smtag.datagen.convert2th -c 191012 -f 10X_L1200_noisy_article_embeddings_unet_32 -X10 -L1200 -E ".//sd-panel" \
    --corrupt ".//sd-tag"


Anonymized geneproduct, except reporters:

    python -m smtag.datagen.convert2th -c 191012 -f 10X_L1200_anonym_not_reporter_article_embeddings_unet_32 -L1200 -X10 -E ".//sd-panel" -y ".//sd-tag[@type='gene']",".//sd-tag[@type='protein']" -e ".//sd-tag[@type='gene']",".//sd-tag[@type='protein']" -A ".//sd-tag[@role='intervention']",".//sd-tag[@role='assayed']"

Anonymized small molecules:

    python -m smtag.datagen.convert2th -c 191012 -f 10X_L1200_molecule_anonym_article_embeddings_unet_32 -L1200 -X10 -E ".//sd-panel" -e ".//sd-tag[@type='molecule']" -A ".//sd-tag[@role='intervention']",".//sd-tag[@role='assayed']"


Disease brat

    python -m smtag.datagen.convert2th -c NCBI_disease -b -L1200 -X10 -f 10X_L1200_disease_article_embeddings_unet_32

SARS minidataset

    python -m smtag.datagen.convert2th -c 200502SARS -f 10X_L1200_SARS_article_embeddings_unet_32 -X10 -L1200 -E ".//sd-panel


## __figure-level__

Unmodified figure-level captions:

    python -m smtag.datagen.convert2th -c 191012 -f 10X_L1200_figure_article_embeddings_unet_32 -X10 -L1200 -E ".//fig/caption"


Unmodified figure-level noisy:

    _corrupt_proba     = 0.01

    python -m smtag.datagen.convert2th -c 191012 -f 10X_L1200_noisy_figure_article_embeddings_unet_32 -X10 -L1200 -E ".//fig/caption" --corrupt ".//sd-tag"


Limited to early emboj with consistent panel labels:

    python -m smtag.datagen.convert2th -c emboj_until_2012 -f 10X_L1200_figure_emboj_2012_article_embeddings_unet_32 -X10 -L1200 -E ".//fig/caption"


# Models

## Multi-entities with exp assays

### Panel level model:

Training:

    python -m smtag.train.meta -f 10X_L1200_article_embeddings_unet_32 -E100 -Z32 -R0.005 \
    -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay \
    --kernel_table 3,3,3,3 --nf_table 128,128,128,128,128 --stride_table 1,1,1,1 --dropout_rate 0.5

Model: `2020-04-20-07-17_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_008.zip`


__Production model (`--production`): `2020-04-20-10-19_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_006.zip`__

### Figure and panel level model:

Training:

    python -m smtag.train.meta -f 10X_L1200_article_embeddings_unet_32,10X_L1200_figure_article_embeddings_unet_32 -E40 -Z32 -R0.001 -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay --kernel_table 3,3,3 --nf_table 64,128,128,128 --stride_table 1,1,1 --dropout_rate 0.2

Model: `2020-04-21-13-31_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_004.zip`


__Production model (`--production`): `2020-04-21-14-55_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_004.zip`__


### Model with noisy datasets:

    python -m smtag.train.meta -f 10X_L1200_noisy_article_embeddings_unet_32,10X_L1200_noisy_figure_article_embeddings_unet_32 -E200 -Z32 -R0.0001 -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay --kernel_table 3,3,3 --nf_table 64,128,128,128 --stride_table 1,1,1 --dropout_rate 0.2

model: `2020-04-29-15-34_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_025.zip`

__Production model__ (`--production`): `2020-04-30-00-42_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_020.zip `


### Model with noisy datasets and SARS minidataset:

    python -m smtag.train.meta -f 10X_L1200_noisy_article_embeddings_unet_32,10X_L1200_noisy_figure_article_embeddings_unet_32,10X_L1200_SARS_article_embeddings_unet_32 -E200 -Z32 -R0.0001 -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay --kernel_table 3,3,3 --nf_table 64,128,128,128 --stride_table 1,1,1 --dropout_rate 0.2

__Production model__ (`--production`): `2020-05-02-16-45_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_020.zip`

### Model with only figure-level noisy and SARS minidataset:

    python -m smtag.train.meta -f 10X_L1200_noisy_figure_article_embeddings_unet_32,10X_L1200_SARS_article_embeddings_unet_32 -E200 -Z32 -R0.0001 -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay --kernel_table 3,3,3 --nf_table 64,128,128,128 --stride_table 1,1,1 --dropout_rate 0.2 --production

__Production model__ (`--production`): `2020-05-04-16-52_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_040.zip`

## Roles for gene products:

Training:

    python -m smtag.train.meta -f 10X_L1200_anonym_not_reporter_article_embeddings_unet_32 -E20 -Z32 -R0.0001 -o intervention,assayed --kernel_table 3,3,3,3 --nf_table 64,128,128,128,128 --stride_table 1,1,1,1 --dropout_rate 0.2

Model: `2020-04-22-09-33_intervention_assayed_epoch_006.zip`

__Production model (`--production`): `2020-04-22-11-12_intervention_assayed_epoch_007.zip`__


## Roles for small molecule

    python -m smtag.train.meta -f 10X_L1200_molecule_anonym_article_embeddings_unet_32 -E20 -Z32 -R0.0001 -o intervention,assayed --kernel_table 3,3,3,3 --nf_table 64,128,128,128,128 --stride_table 1,1,1,1 --dropout_rate 0.2

Model: `2020-04-22-22-16_intervention_assayed_epoch_013.zip`

__Production model (`--production`): `2020-04-23-13-17_intervention_assayed_epoch_022.zip`


## Reporter

Training:

    python -m smtag.train.meta -f 10X_L1200_article_embeddings_unet_32 -E20 -Z32 -R0.0001 -o reporter --kernel_table 3,3,3 --nf_table 64,128,128,128 --stride_table 1,1,1 --dropout_rate 0.2

Model: `2020-04-23-15-54_reporter_epoch_019.zip`


__Production model (`--production`): `2020-04-23-18-58_reporter_epoch_019.zip`__


## Diseases

Training:

_Note: using 10X_L1200_figure_emboj_2012_article_embeddings_unet_32 as 'decoy' dataset to incread number of negative example, since brat dataset very small. Decoy should not be too large or too confusing (eg include only few or no positives)._

    python -m smtag.train.meta -f 10X_L1200_disease_article_embeddings_unet_32,10X_L1200_figure_emboj_2012_article_embeddings_unet_32 -E100 -Z32 -R0.0001 -o disease --kernel_table 3,3,3 --nf_table 64,128,128,128 --stride_table 1,1,1 --dropout_rate 0.2

Model: `2020-04-23-22-20_disease_epoch_060.zip`

__Production model (`--production`): `2020-04-24-07-33_disease_epoch_099.zip`__

With SARS minidataset:

    python -m smtag.train.meta -f 10X_L1200_disease_article_embeddings_unet_32,10X_L1200_figure_emboj_2012_article_embeddings_unet_32,10X_L1200_SARS_article_embeddings_unet_32  -E100 -Z32 -R0.0001 -o disease --kernel_table 3,3,3 --nf_table 64,128,128,128 --stride_table 1,1,1 --dropout_rate 0.2 --production


__Production model (`--production`): `2020-05-02-17-54_disease_epoch_099.zip`__

## Panel segmentation:

Training: 

    python -m smtag.train.meta -f 10X_L1200_figure_emboj_2012_article_embeddings_unet_32 -E20 -Z32 -R0.0001 -o panel_stop --kernel_table 3,3,3 --nf_table 64,128,128,128 --stride_table 1,1,1 --dropout_rate 0.2

Model: `2020-04-24-09-43_panel_stop_epoch_099.zip`

__Production model (`--production`): `2020-04-24-11-24_panel_stop_epoch_099.zip`__


# Benchmarking

Entities on non-noisy dataset:

    python -m smtag.train.evaluator 10X_L1200_article_embeddings_unet_32,10X_L1200_figure_article_embeddings_unet_32 2020-04-21-13-31_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_004.zip

Entities on noise-added dataset:

    python -m smtag.train.evaluator 10X_L1200_noisy_article_embeddings_unet_32,10X_L1200_noisy_figure_article_embeddings_unet_32 2020-04-30-00-42_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_020.zip

Geneprod roles:

    python -m smtag.train.evaluator 10X_L1200_anonym_not_reporter_article_embeddings_unet_32 2020-04-22-11-12_intervention_assayed_epoch_007.zip

Small mol roles:

    python -m smtag.train.evaluator 10X_L1200_molecule_anonym_article_embeddings_unet_32 2020-04-23-13-17_intervention_assayed_epoch_022.zip

Reporter:

    python -m smtag.train.evaluator 10X_L1200_article_embeddings_unet_32 2020-04-23-18-58_reporter_epoch_019.zip

Diseases:

    python -m smtag.train.evaluator 10X_L1200_disease_article_embeddings_unet_32,10X_L1200_figure_emboj_2012_article_embeddings_unet_32 2020-04-24-07-33_disease_epoch_099.zip

Panels:

    python -m smtag.train.evaluator 10X_L1200_figure_emboj_2012_article_embeddings_unet_32 2020-04-24-11-24_panel_stop_epoch_099.zip





======================================================== 

 Data: /workspace/py-smtag/resources/data4th/10X_L1200_article_embeddings_unet_32 + /workspace/py-smtag/resources/data4th/10X_L1200_figure_article_embeddings$
unet_32

 Model: 2020-04-21-13-31_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_004.zip 
 
 Global stats:
 
        precision = 0.8195339441299438
        recall = 0.78
        f1 = 0.80

 Feature: 'small_molecule'

        precision = 0.76
        recall = 0.73
        f1 = 0.74

 Feature: 'geneprod'

        precision = 0.87
        recall = 0.87
        f1 = 0.87

 Feature: 'subcellular'

        precision = 0.76
        recall = 0.72
        f1 = 0.74

 Feature: 'cell'

        precision = 0.83
        recall = 0.84
        f1 = 0.83

Feature: 'tissue'

        precision = 0.74
        recall = 0.69
        f1 = 0.72

 Feature: 'organism'

        precision = 0.78
        recall = 0.77
        f1 = 0.77

 Feature: 'assay'

        precision = 0.82
        recall = 0.66
        f1 = 0.73

 Feature: 'untagged'

        precision = 0.99
        recall = 0.99
        f1 = 0.99


========================================================

 Data: /workspace/py-smtag/resources/data4th/10X_L1200_noisy_article_embeddings_unet_32 + /workspace/py-smtag/resources/data4th/10X_L1200_noisy_figure_article
_embeddings_unet_32

 Model: 2020-04-30-00-42_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_020.zip

Global stats: 

        precision = 0.8328481912612915
        recall = 0.78
        f1 = 0.80

 Feature: 'small_molecule'

        precision = 0.77
        recall = 0.74
        f1 = 0.75

 Feature: 'geneprod'

        precision = 0.88
        recall = 0.88
        f1 = 0.88

 Feature: 'subcellular'

        precision = 0.78
        recall = 0.71
        f1 = 0.74

 Feature: 'cell'

        precision = 0.87
        recall = 0.82
        f1 = 0.84

 Feature: 'tissue'

        precision = 0.77
        recall = 0.66
        f1 = 0.71

 Feature: 'organism'

        precision = 0.79
        recall = 0.75
        f1 = 0.77

 Feature: 'assay'

        precision = 0.82
        recall = 0.66
        f1 = 0.73

 Feature: 'untagged'

        precision = 0.99
        recall = 1.00
        f1 = 0.99


========================================================

 Data: /workspace/py-smtag/resources/data4th/10X_L1200_anonym_not_reporter_article_embeddings_unet_32

 Model: 2020-04-22-11-12_intervention_assayed_epoch_007.zip

 Global stats: 

        precision = 0.9069655537605286
        recall = 0.91
        f1 = 0.91

 Feature: 'intervention'

        precision = 0.83
        recall = 0.85
        f1 = 0.84

 Feature: 'assayed'

        precision = 0.89
        recall = 0.88
        f1 = 0.88

 Feature: 'untagged'

        precision = 1.00
        recall = 1.00
        f1 = 1.00


========================================================

 Data: /workspace/py-smtag/resources/data4th/10X_L1200_molecule_anonym_article_embeddings_unet_32

 Model: 2020-04-23-13-17_intervention_assayed_epoch_022.zip

 Global stats: 

        precision = 0.9329233169555664
        recall = 0.91
        f1 = 0.92

 Feature: 'intervention'

        precision = 0.95
        recall = 0.97
        f1 = 0.96

 Feature: 'assayed'

        precision = 0.85
        recall = 0.75
        f1 = 0.80

 Feature: 'untagged'

        precision = 1.00
        recall = 1.00
        f1 = 1.00


========================================================

 Data: /workspace/py-smtag/resources/data4th/10X_L1200_article_embeddings_unet_32

 Model: 2020-04-23-18-58_reporter_epoch_019.zip

 Global stats:

        precision = 0.9218883514404297
        recall = 0.90
        f1 = 0.91

 Feature: 'reporter'

        precision = 0.84
        recall = 0.79
        f1 = 0.82

 Feature: 'untagged'

        precision = 1.00
        recall = 1.00
        f1 = 1.00

========================================================

 Data: /workspace/py-smtag/resources/data4th/10X_L1200_disease_article_embeddings_unet_32 + /workspace/py-smtag/resources/data4th/10X_L1200_figure_emboj_2012_article_embeddings_unet_32

 Model: 2020-04-24-07-33_disease_epoch_099.zip

 Global stats: 

        precision = 0.9428324699401855
        recall = 0.93
        f1 = 0.94

 Feature: 'disease'

        precision = 0.89
        recall = 0.86
        f1 = 0.87

 Feature: 'untagged'

        precision = 0.99
        recall = 1.00
        f1 = 1.00


========================================================

 Data: /workspace/py-smtag/resources/data4th/10X_L1200_figure_emboj_2012_article_embeddings_unet_32

 Model: 2020-04-24-11-24_panel_stop_epoch_099.zip

 Global stats: 

        precision = 0.9986858367919922
        recall = 1.00
        f1 = 1.00

 Feature: 'panel_stop'

        precision = 1.00
        recall = 0.99
        f1 = 1.00

 Feature: 'untagged'

        precision = 1.00
        recall = 1.00
        f1 = 1.00
