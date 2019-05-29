# Preparation of corpus of xml documents and images

Required data:
- latest sd-graph in neo4j.
- NCBI_disease corpus in brat format.

    smtag-neo2xml -l10000 -f 181203all # on large sd-graph with 30000 panels from 1100 papers
    smtag-neo2xml -y 1998:2012 -J "embo journal" -f emboj_until_2012 -l 10000
    smtag-ocr # run only once
    smtag-viz # run only once
    
# Preparation of ready-to-train datasets

    smtag-convert2th -c 190414 -f 5X_L1200_fig -X5 -L1200 -E ".//fig/caption" -w /ebs/smtag
    smtag-convert2th -c 190414 -L1200 -X5 -E ".//fig/caption" -y ".//sd-tag[@type='gene']",".//sd-tag[@type='protein']" -e ".//sd-tag[@type='gene']",".//sd-tag[@type='protein']" -A ".//sd-tag[@role='intervention']",".//sd-tag[@role='assayed']",".//sd-tag[@role='normalizing']",".//sd-tag[@role='experiment']",".//sd-tag[@role='component']" -f 5X_L1200_geneprod_anonym_not_reporter_fig -w /ebs/smtag
    smtag-convert2th -c 190414 -L1200 -X5 -E ".//fig/caption" -e ".//sd-tag[@type='molecule']" -A ".//sd-tag[@role='intervention']",".//sd-tag[@role='assayed']" -f 5X_L1200_molecule_anonym_fig -w /ebs/smtag
    smtag-convert2th -c NCBI_disease -b -L1200 -X10 -f 10X_L1200_disease -w /ebs/smtag
    smtag-convert2th -c emboj_until_2012 -f 5X_L1200_emboj_2012_no_viz -X5 -L1200 -E ".//fig/caption" --noviz --noocr -w /ebs/smtag

# Models with viz context

Use `-V 500` to include 500 features from visual context.

## Multi-entities with exp assays and viz:

    smtag-meta -f 5X_L1200_fig -E60 -Z64 -R0.0001 -D0.2 -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay -k 6,6,6 -n 32,64,128 -p 2,2,2 -V 500 -w /ebs/smtag
    
<img src="figures/.png" width="50%">
__Model: `...`__

# Roles geneprod with viz:

    smtag-meta -f 5X_L1200_geneprod_anonym_not_reporter_fig -E60 -Z64 -R0.0001 -D0.2 -o intervention,assayed -k 6,6,6 -n 32,64,128 -p 2,2,2 -V 500 -w /ebs/smtag
 
<img alt="plots" src="figures/.png" width="500">
__Model: `...`__
    
# Role for small molecule with viz:

    smtag-meta -f 5X_L1200_molecule_anonym_fig -E60 -Z64 -R0.0001 -D0.2 -o intervention,assayed -k 6,6,6 -n 32,64,128 -p 2,2,2 -V 500 -w /ebs/smtag
    
<img src="figures/.png" width="50%">
__Model: `...`__


# Models without viz

No `-V` option.

## Multi entities with exp assays and __without__ viz context:

    smtag-meta -f 5X_L1200_fig -E60 -Z64 -R0.001 -D0.2 -o small_molecule,geneprod,subcellular,cell,tissue,organism,assay -k 6,6,6 -n 32,64,128 -p 2,2,2 -w /ebs/smtag
    
<img src="figures/.png" width="50%">
__Model: `...`__

## Geneprod roles __without__ viz:

    smtag-meta -f 5X_L1200_geneprod_anonym_not_reporter_fig -E60 -Z64 -R0.0001 -D0.2 -o intervention,assayed -k 6,6,6 -n 32,64,128 -p 2,2,2 -w /ebs/smtag
    
<img src="figures/.png" width="500px">
__Model: `...`__

## Role for small molecule __without__ viz context (faster learning rate):

    smtag-meta -f 5X_L1200_molecule_anonym_fig -E60 -Z64 -R0.001 -D0.2 -o intervention,assayed -k 6,6,6 -n 32,64,128 -p 2,2,2 -w /ebs/smtag
    
<img src="figures/.png" width="50%">
__Model: `...`__


## Reporter __without__ viz:

    smtag-meta -f 5X_L1200_fig -E60 -Z64 -R0.001 -o reporter -k 6,6,6 -n 32,64,128 -p 2,2,2 -w /ebs/smtag
    
<img src="figures/.png" width="500px">
Model: __`..`__

## Disease __without__ viz context:

    smtag-meta -f 10X_L1200_disease,5X_L1200_fig -E120 -Z64 -R0.0001 -D0.2 -o disease -k 6,6,6 -n 32,64,128 -p 2,2,2 -w /ebs/smtag
        
<img src="figures/.png" width="50%">
__Model: `...`__

## Panels on emboj only:

    smtag-meta -f 5X_L1200_emboj_2012_no_viz -E120 -Z64 -R0.001 -o panel_stop -k 6,6,6 -n 32,64,128 -p 2,2,2
    
<img src="figures/.png" width="50%">
__Model: `...`__

