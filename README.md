# SecretoGen


A conditional autoregressive model of signal peptides. Work in progress.


## Resources

The trained model weights are available at https://sid.erda.dk/cgi-sid/ls.py?share_id=e9OpPDduHg.


## Overview

The SecretoGen model architecture is defined in `src/pretraining/models/seq2seq.py`. The taxonomy vocabulary is stored in `data/taxonomy_mappings`. We use this for mapping a species to its set of taxonomy tokens using `src.utils.alphabets.TaxonomyMapping`.

## Computing perplexities

The script `src/compute_perplexity.py` can score SecretoGen perplexities from tsv-formatted input data.


## Baseline perplexities

`compute_perplexity_progen.py` and `compute_perplexity_spgen.py` work on the same input format.
For SPGen, you will have to edit the checkpoint paths on lines 23 to 26. Checkpoints were prepared by extracting the state dicts from the checkpoints in the original SPGen repository, so that loading the checkpoint files does no longer depend on the SPGen directory structure for dependencies.