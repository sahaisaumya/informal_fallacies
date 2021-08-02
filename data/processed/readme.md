### Folders

This is the processed dataset that works with accompanying [code](https://github.com/sahaisaumya/informal_fallacies/tree/main/code).

`annotated_dataset` : data with samples annotated by crowd-workers  
`full_dataset` : data with samples annotated by workers and negative samples as described in section 4 of the paper.

Each folder has train.txt, dev.txt and test.txt (stratified split 70-20-10) that were used for training, validation and testing respectively 

---
### Data description

Each data file has tab separated with following columns: 

* Column 1: Tokenized and cleaned comment of interest (COI) -> list 
* Column 2: Original COI -> string 
* Column 3: Binary (fallacy/non_fallacy) token wise class of COI tokens of column 1 -> list 
* Column 4: Multi-class (one of 8 fallacy classes or none) token wise class of COI tokens of column 1 -> list 
* Column 5: Parent of COI -> string 
* Column 6: Title of the submission -> string 
* Column 7: Label comment -> string 
