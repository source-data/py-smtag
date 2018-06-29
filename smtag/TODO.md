- implement GPU computing: DONE
- implement assertEqualTensor properly: DONE
- change Binarize Predictor to TString: DONE
- create TString class to give string behavior to encoded string: DONE
- finish options for meta (collapse, overlap, OR tables): DONE
- revise anonymization and use OR gate instead of label character: DONE
- test other anonymization replacement characters/patters: 0000 and FFFF: DONE could be that 0000 is the best
- add 'cartridges' to load set of models: DONE

----

- make venv work on GPU
- SmtagEngine classes to combine entity, pure context, context-text, boundary models
- test py-smtag on GPU
- test datapret with brat format
- implement/test model on boundaries (instead of longitudinal marks)
- train all models (all entities, panel_start, context_geneprod, context_small_mol, disease, exp_assay)
- connect Flask server

- implement -af option in sdgraph2th to concatenate all figures for a single paper
- enable tensorboardX on Amazon GPU machine (serve which URL? how to set security?)
- restructure smtag package in subpackages: datagen/, train/, predict/, common/ ?
- add pydoc comments on all classes and check commenting style
- use logging with different verbosity levels
- check usage of dtype = uint8 (ByteTensor) for binarized tensors when generating/saving/loading datasets and in Binarized
- update README.md
- move Converter.t_encode(example) from loader to dataprep to save datatasets as 4 file bundles
- pre-tokenize when generating datasets
- change sdgraph2neo to save xml files and XMLFeaturizer to read file from dir with direct application to learn JATS
- accuracy and benchmarking classes
- implement Hinton's capsules

