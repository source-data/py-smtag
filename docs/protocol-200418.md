# Models based on UNET (19 April 20202)

Using the ai@dev branch, vesearch@dev and py-smtag@unet branches

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

    python -m vsearch.train datasets/oapmc_abstracts/ -E40 -Z32 -R0.001 --nn unet # --> recall=0.88, f1=0.93 save as: 2020-04-18-23-03_final.zip

    cp models/2020-04-18-23-03_final.zip ../py-smtag/rack/2020-04-18-23-03_final.zip




# Preparation of ready-to-train datasets

## __panel-level__

Unmodified panel-level captions:

    python -m smtag.datagen.convert2th -c 191012 -f 10X_L1200_article_embeddings_unet_32 -X10 -L1200 -E ".//sd-panel"


Unmodified panel-level captions with noise:

    _corrupt_path      = ".//sd-tag"
    _corrupt_proba     = 0.1

    python -m smtag.datagen.convert2th -c 191012 -f 10X_L1200_noisy_article_embeddings_unet_32 -X10 -L1200 -E ".//sd-panel"

Anonymized geneproduct, except reporters:

    python -m smtag.datagen.convert2th -c 191012 -f 10X_L1200_anonym_not_reporter_article_embeddings_unet_32 -L1200 -X10 -E ".//sd-panel" -y ".//sd-tag[@type='gene']",".//sd-tag[@type='protein']" -e ".//sd-tag[@type='gene']",".//sd-tag[@type='protein']" -A ".//sd-tag[@role='intervention']",".//sd-tag[@role='assayed']"

Anonymized small molecules:

    python -m smtag.datagen.convert2th -c 191012 -f 10X_L1200_molecule_anonym_article_embeddings_unet_32 -L1200 -X10 -E ".//sd-panel" -e ".//sd-tag[@type='molecule']" -A ".//sd-tag[@role='intervention']",".//sd-tag[@role='assayed']"

Disease brat

    python -m smtag.datagen.convert2th -c NCBI_disease -b -L1200 -X10 -f 10X_L1200_disease_article_embeddings_unet_32


## __figure-level__

Unmodified figure-level captions:

    python -m smtag.datagen.convert2th -c 191012 -f 10X_L1200_figure_article_embeddings_unet_32 -X10 -L1200 -E ".//fig/caption"

Limited to early emboj with consistent panel labels:

    python -m smtag.datagen.convert2th -c emboj_until_2012 -f 10X_L1200_figure_emboj_2012_article_embeddings_unet_32 -X10 -L1200 -E ".//fig/caption"


# Models

All trained on the `@unet` branch.

## Multi-entities with exp assays

### Panel level model:

Training:

    python -m smtag.train.meta -f 10X_L1200_article_embeddings_unet_32 -E100 -Z32 -R0.005 \
    -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay \
    --kernel_table 3,3,3,3 --nf_table 128,128,128,128,128 --stride_table 1,1,1,1 --dropout_rate 0.5

Model: `2020-04-20-07-17_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_008.zip`


Benchmarking:

    python -m smtag.train.evaluator 10X_L1200_article_embeddings_unet_32 

__Production model (`--production`): `2020-04-20-10-19_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_006.zip`__

### Figure and panel level model:

Training:

    python -m smtag.train.meta -f 10X_L1200_article_embeddings_unet_32,10X_L1200_figure_article_embeddings_unet_32 -E40 -Z32 -R0.001 -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay --kernel_table 3,3,3 --nf_table 64,128,128,128 --stride_table 1,1,1 --dropout_rate 0.2

Model: `2020-04-21-13-31_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_004.zip`


Benchmarking:


__Production model (`--production`): `2020-04-21-14-55_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_004.zip`__


## Roles for gene products:

Training:

    python -m smtag.train.meta -f 10X_L1200_anonym_not_reporter_article_embeddings_unet_32 -E20 -Z32 -R0.0001 -o intervention,assayed --kernel_table 3,3,3,3 --nf_table 64,128,128,128,128 --stride_table 1,1,1,1 --dropout_rate 0.2

Model: `2020-04-22-09-33_intervention_assayed_epoch_006.zip`

Benchmarking:

    python -m smtag.train.evaluator  2020-02-28-09-42_intervention_assayed_epoch_017.zip

__Production model (`--production`): `2020-04-22-11-12_intervention_assayed_epoch_007.zip`__


## Roles for small molecule

    python -m smtag.train.meta -f 10X_L1200_molecule_anonym_article_embeddings_unet_32 -E20 -Z32 -R0.0001 -o intervention,assayed --kernel_table 3,3,3,3 --nf_table 64,128,128,128,128 --stride_table 1,1,1,1 --dropout_rate 0.2

Model: `2020-04-22-22-16_intervention_assayed_epoch_013.zip`

__Production model (`--production`): `2020-04-23-13-17_intervention_assayed_epoch_022.zip`



## Reporter

Training:

    python -m smtag.train.meta -f 10X_L1200_article_embeddings_unet_32 -E20 -Z32 -R0.0001 -o reporter --kernel_table 3,3,3 --nf_table 64,128,128,128 --stride_table 1,1,1 --dropout_rate 0.2

Model: `2020-04-23-15-54_reporter_epoch_019.zip`

Benchmarking:

    python -m smtag.train.evaluator 

__Production model (`--production`): `2020-04-23-18-58_reporter_epoch_019.zip`__


## Diseases

Training:

_Note: using 10X_L1200_figure_emboj_2012_article_embeddings_unet_32 as 'decoy' dataset to incread number of negative example, since brat dataset very small. Decoy should not be too large or too confusing (eg include only few or no positives)._

    python -m smtag.train.meta -f 10X_L1200_disease_article_embeddings_unet_32,10X_L1200_figure_emboj_2012_article_embeddings_unet_32 -E100 -Z32 -R0.0001 -o disease --kernel_table 3,3,3 --nf_table 64,128,128,128 --stride_table 1,1,1 --dropout_rate 0.2

Model: `2020-04-23-22-20_disease_epoch_060.zip`

Benchmarking:

    python -m smtag.train.evaluator 

__Production model (`--production`): `2020-04-24-07-33_disease_epoch_099.zip`__


## Panel segmentation:

Training: 

    python -m smtag.train.meta -f 10X_L1200_figure_emboj_2012_article_embeddings_unet_32 -E20 -Z32 -R0.0001 -o panel_stop --kernel_table 3,3,3 --nf_table 64,128,128,128 --stride_table 1,1,1 --dropout_rate 0.2


Model: `2020-04-24-09-43_panel_stop_epoch_099.zip`

Benchmarking:

    python -m smtag.train.evaluator 

__Production model (`--production`): `2020-04-24-11-24_panel_stop_epoch_099.zip`__

