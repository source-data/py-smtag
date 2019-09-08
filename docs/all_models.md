# Preparation of corpus of xml documents and images

Required data:

- latest sd-graph in neo4j.
- NCBI_disease corpus in brat format.

Generation of corpuses from sd-graph:

    smtag-neo2xml -l10000 -f 181203all # on large sd-graph with 30000 panels from 1100 papers
    smtag-neo2xml -Y 1998:2012 -J "embo journal" -f emboj_until_2012 -l 10000

Processing of images:

    # smtag-ocr # run only once, optional since not yet in use
    smtag-viz # run only once
    
# Preparation of ready-to-train datasets, panel level

    smtag-convert2th -c 190414 -f 5X_L1200_article_embeddings_128 -X5 -L1200 -E ".//sd-panel" --noocr --noviz

    smtag-convert2th -c 190414 -f 5X_L1200_anonym_not_reporter_article_embeddings_128 -L1200 -X5 -E ".//sd-panel" -y ".//sd-tag[@type='gene']",".//sd-tag[@type='protein']" -e ".//sd-tag[@type='gene']",".//sd-tag[@type='protein']" -A ".//sd-tag[@role='intervention']",".//sd-tag[@role='assayed']" --noocr --noviz
    
    smtag-convert2th -c 190414 -L1200 -X5 -E ".//sd-panel" -e ".//sd-tag[@type='molecule']" -A ".//sd-tag[@role='intervention']",".//sd-tag[@role='assayed']" -f 5X_L1200_molecule_anonym_article_embeddings_128 -p rack --noocr --noviz
    
    smtag-convert2th -c NCBI_disease -b -L1200 -X10 -f 10X_L1200_disease_articke_embeddings_128 --noviz --noocr

    smtag-convert2th -c emboj_until_2012 -f 5X_L1200_emboj_2012_no_viz -X5 -L1200 -E ".//fig/caption" --noviz --noocr
    


# Models without viz

No `-V` option.

## Multi entities with exp assays and __without__ viz context:

    smtag-meta -f 5X_L1200_article_embeddings_128 -E20 -Z32 -R0.005 -D0.2 -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3

    5X_L1200_article_embeddings_128_reproduce_noviz_noocr
    
<img src="figures/.png" width="50%">
__Model: `5X_L1200_article_embeddings_128_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_2019-08-23-17-46.zip`__

Benchmarking:

    python -m smtag.train.evaluator 5X_L1200_article_embeddings_128 5X_L1200_article_embeddings_128_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_2019-08-23-17-46.zip


## Geneprod roles __without__ viz:

    smtag-meta -f 5X_L1200_anonym_not_reporter_article_embeddings_128 -E14 -Z32 -R0.005 -D0.2 -o intervention,assayed -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3
    
<img src="figures/.png" width="500px">
__Model: `5X_L1200_anonym_not_reporter_article_embeddings_128_intervention_assayed_2019-08-22-16-25.zip`__

Benchmarking

    python -m smtag.train.evaluator 5X_L1200_anonym_not_reporter_article_embeddings_128 5X_L1200_anonym_not_reporter_article_embeddings_128_intervention_assayed_2019-08-22-16-25.zip

## Role for small molecule __without__ viz context (faster learning rate):

    smtag-meta -f 5X_L1200_molecule_anonym_article_embeddings_128 -E20 -Z64 -R0.001 -D0.2 -o intervention,assayed -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3
    
<img src="figures/.png" width="50%">
__Model: `5X_L1200_molecule_anonym_article_embeddings_128_intervention_assayed_2019-08-28-23-33_epoch_51.zip`__

Benchmarking:

    python -m smtag.train.evaluator 5X_L1200_molecule_anonym_article_embeddings_128 5X_L1200_molecule_anonym_article_embeddings_128_intervention_assayed_2019-08-28-23-33_epoch_51.zip

## Reporter __without__ viz:

    smtag-meta -f 5X_L1200_article_embeddings_128 -E60 -Z64 -R0.001 -o reporter -k 7,7,7 -n 32,64,128 -g 3,3,3
    
<img src="figures/.png" width="500px">
Model: `5X_L1200_article_embeddings_128_epoch_23.zip` renamed __`5X_L1200_article_embeddings_128_reporter_2019-08-28-00-08_epoch_23_.zip`__

Benchmarking:

    python -m smtag.train.evaluator 5X_L1200_article_embeddings_128 5X_L1200_article_embeddings_128_reporter_2019-08-28-00-08_epoch_23_.zip

## Disease __without__ viz context:

    smtag-meta -f 10X_L1200_disease_articke_embeddings_128,5X_L1200_article_embeddings_128 -E40 -Z32 -R0.001 -D0.25 -o disease -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3
        
<img src="figures/.png" width="50%">
__Model: `10X_L1200_disease_articke_embeddings_128-5X_L1200_article_embeddings_128_disease_2019-08-25-21-47.zip`__

Benchmarking:

    #python -m smtag.train.evaluator 10X_L1200_disease_articke_embeddings_128,5X_L1200_article_embeddings_128 10X_L1200_disease_articke_embeddings_128-5X_L1200_article_embeddings_128_disease_2019-08-25-21-47.zip

## Panels on emboj only:

    smtag-meta -f 5X_L1200_emboj_2012_no_viz -E20 -Z64 -R0.001 -o panel_stop -k 7,7,7,7,7,7,7,7,7,7 -n 128,128,128,128,128,128,128,128,128,128 -g 3,3,3,3,3,3,3,3,3,3

<img src="figures/.png" width="50%">
__Model: `5X_L1200_emboj_2012_no_viz_panel_stop_2019-08-29-08-31.zip`__


Benchmarking:
    
    python -m smtag.train.evaluator 5X_L1200_emboj_2012_no_viz 5X_L1200_emboj_2012_no_viz_panel_stop_2019-08-29-08-31.zip
 


