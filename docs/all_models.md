# Preparation of corpus of xml documents and images

Required data:

- latest sd-graph in neo4j.
- NCBI_disease corpus in brat format.

Generation of corpuses from sd-graph:

    smtag-neo2xml -l10000 -f 191012 -p rack # on large sd-graph with 30000 panels from 1100 papers
    smtag-neo2xml -Y 1998:2012 -J "embo journal" -f emboj_until_2012 -l 10000

Processing of images:

    # smtag-ocr # run only once, optional since not yet in use
    smtag-viz # run only once

# Preparation of ready-to-train datasets, panel level

<<<<<<< HEAD
    smtag-convert2th -c 190414 -f 5X_L1200_unet_no_embed -X5 -L1200 -E ".//sd-panel" -p rack

    smtag-convert2th -c 190414 -f 5X_L1200_anonym_not_reporter_unet_no_embed -L1200 -X5 -E ".//sd-panel" -y ".//sd-tag[@type='gene']",".//sd-tag[@type='protein']" -e ".//sd-tag[@type='gene']",".//sd-tag[@type='protein']" -A ".//sd-tag[@role='intervention']",".//sd-tag[@role='assayed']"
    
    smtag-convert2th -c 190414 -L1200 -X5 -E ".//sd-panel" -e ".//sd-tag[@type='molecule']" -A ".//sd-tag[@role='intervention']",".//sd-tag[@role='assayed']" -f 5X_L1200_molecule_anonym_article_embeddings_128 -p rack --noocr --noviz
    
    smtag-convert2th -c NCBI_disease -b -L1200 -X10 -f 10X_L1200_disease_articke_embeddings_128 --noviz --noocr
=======
    smtag-convert2th -c 191012 -f 10X_L1200_article_embeddings_128 -X10 -L1200 -E ".//sd-panel" --noocr --noviz -p rack

    smtag-convert2th -c 191012 -f 10X_L1200_anonym_not_reporter_article_embbeddings_128 -L1200 -X10 -E ".//sd-panel" -y ".//sd-tag[@type='gene']",".//sd-tag[@type='protein']" -e ".//sd-tag[@type='gene']",".//sd-tag[@type='protein']" -A ".//sd-tag[@role='intervention']",".//sd-tag[@role='assayed']" --noocr --noviz
>>>>>>> dev

    smtag-convert2th -c 191012 -f 10X_L1200_molecule_anonym_article_embeddings_128 -L1200 -X10 -E ".//sd-panel" -e ".//sd-tag[@type='molecule']" -A ".//sd-tag[@role='intervention']",".//sd-tag[@role='assayed']" -p rack --noocr --noviz

    smtag-convert2th -c NCBI_disease -b -L1200 -X10 -f 10X_L1200_disease_article_embeddings_128 --noviz --noocr -p rack

    smtag-convert2th -c 191012 -f 10X_L1200_figure_article_embeddings_128 -X10 -L1200 -E ".//fig/caption" --noocr --noviz -p rack
    smtag-convert2th -c emboj_until_2012 -f 10X_L1200_figure_emboj_2012_article_embeddings_128 -X10 -L1200 -E ".//fig/caption" --noviz --noocr



# Models without viz

No `-V` option.

## Multi entities with exp assays and __without__ viz context:

    smtag-meta -f 10X_L1200_article_embeddings_128 -E20 -Z32 -R0.005 -D0.2 -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3 -p rack

__Model: `10X_L1200_article_embeddings_128_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_2019-10-13-22-19.zip`__

Benchmarking:

    python -m smtag.train.evaluator 10X_L1200_article_embeddings_128 10X_L1200_article_embeddings_128_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_2019-10-13-22-19.zip

Production model:

    smtag-meta -f 10X_L1200_article_embeddings_128 -E12 -Z32 -R0.005 -D0.2 -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3 -p rack --production

__Production model: `10X_L1200_article_embeddings_128_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_2019-10-27-09-28.zip`__

## Geneprod roles __without__ viz:

<<<<<<< HEAD
    smtag-meta -f 5X_L1200_unet_no_embed -E120 -Z32 -R0.001 -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay -k 7,7,7 -n 128,128,128 --pool_table 2,2,2
=======
    smtag-meta -f 10X_L1200_anonym_not_reporter_article_embbeddings_128 -E30 -Z32 -R0.001 -D0.2 -o intervention,assayed -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3 -p rack
>>>>>>> dev
    
__Model: `10X_L1200_anonym_not_reporter_article_embbeddings_128_intervention_assayed_2019-10-15-13-49_epoch_6.zip`__

without reporter trick:



Benchmarking

    python -m smtag.train.evaluator 10X_L1200_anonym_not_reporter_article_embbeddings_128 10X_L1200_anonym_not_reporter_article_embbeddings_128_intervention_assayed_2019-10-15-13-49_epoch_6.zip

Production:

    smtag-meta -f 10X_L1200_anonym_not_reporter_article_embbeddings_128 -E6 -Z32 -R0.001 -D0.2 -o intervention,assayed -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3 -p rack --production

__Production model: `10X_L1200_anonym_not_reporter_article_embbeddings_128_intervention_assayed_2019-10-27-12-25.zip`__


## Role for small molecule __without__ viz context (faster learning rate):

    smtag-meta -f 10X_L1200_molecule_anonym_article_embeddings_128 -E20 -Z64 -R0.001 -D0.2 -o intervention,assayed -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3
    
__Model: `10X_L1200_molecule_anonym_article_embeddings_128_intervention_assayed_2019-10-18-01-00_epoch_29.zip`__

Benchmarking:

    python -m smtag.train.evaluator 10X_L1200_molecule_anonym_article_embeddings_128 10X_L1200_molecule_anonym_article_embeddings_128_intervention_assayed_2019-10-18-01-00_epoch_29.zip

Production:

    smtag-meta -f 10X_L1200_molecule_anonym_article_embeddings_128 -E23 -Z64 -R0.001 -D0.2 -o intervention,assayed -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3 --production -p rack

__Production model: `10X_L1200_molecule_anonym_article_embeddings_128_intervention_assayed_2019-10-27-21-03.zip`__


## Reporter __without__ viz:

    smtag-meta -f 10X_L1200_article_embeddings_128 -E60 -Z64 -R0.001 -o reporter -k 7,7,7 -n 32,64,128 -g 3,3,3
    

Model: __`10X_L1200_article_embeddings_128_reporter_2019-10-19-00-34.zip`__

Benchmarking:

    python -m smtag.train.evaluator 10X_L1200_article_embeddings_128 10X_L1200_article_embeddings_128_reporter_2019-10-19-00-34.zip

Production:

    smtag-meta -f 10X_L1200_article_embeddings_128 -E40 -Z64 -R0.001 -o reporter -k 7,7,7 -n 32,64,128 -g 3,3,3 --production -p rack

__Production model: `10X_L1200_article_embeddings_128_reporter_2019-10-28-00-43.zip`__


## Disease __without__ viz context:

    smtag-meta -f 10X_L1200_disease_article_embeddings_128,10X_L1200_figure_article_embeddings_128 -E40 -Z32 -R0.001 -D0.25 -o disease -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3 -p rack
        
__Model: `10X_L1200_disease_article_embeddings_128-10X_L1200_figure_article_embeddings_128_disease_2019-10-20-09-19.zip`__

Benchmarking:

    python -m smtag.train.evaluator 10X_L1200_disease_article_embeddings_128,10X_L1200_figure_article_embeddings_128 10X_L1200_disease_article_embeddings_128-10X_L1200_figure_article_embeddings_128_disease_2019-10-20-09-19.zip

Production:

    smtag-meta -f 10X_L1200_disease_article_embeddings_128,10X_L1200_figure_article_embeddings_128 -E40 -Z32 -R0.001 -D0.25 -o disease -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3 -p rack ---production

__Production model: `10X_L1200_disease_article_embeddings_128-10X_L1200_figure_article_embeddings_128_disease_2019-10-28-06-44_epoch_10.zip`__


## General panel segmentation:

    smtag-meta -f  10X_L1200_figure_article_embeddings_128 -E15 -Z32 -R0.001 -o panel_stop -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3 -p rack

__Model: `10X_L1200_figure_article_embeddings_128_panel_stop_2019-10-16-06-31.zip`__

## Panels on emboj only:

    smtag-meta -f 10X_L1200_figure_emboj_2012_article_embeddings_128 -E20 -Z64 -R0.001 -o panel_stop -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3 -p rack

__Model: `10X_L1200_figure_emboj_2012_article_embeddings_128_panel_stop_2019-10-20-14-09.zip`__


Benchmarking:

    python -m smtag.train.evaluator 10X_L1200_figure_emboj_2012_article_embeddings_128 10X_L1200_figure_emboj_2012_article_embeddings_128_panel_stop_2019-10-20-14-09.zip

Production:

    smtag-meta -f 10X_L1200_figure_emboj_2012_article_embeddings_128 -E20 -Z64 -R0.001 -o panel_stop -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3 -p rack --production

__Production model: `10X_L1200_figure_emboj_2012_article_embeddings_128_panel_stop_2019-10-28-11-19.zip`__
