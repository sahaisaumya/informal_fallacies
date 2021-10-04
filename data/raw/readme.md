#### Raw data

Unprocessed data for each of the considered 8 fallacies. <br>
`*_annotated.csv` : Data collected from crowd-sourcing <br>
`*_negative-samples.csv` : Only the sampled negative samples (as described in section 4)

---

#### Metadata for `*_annotated.csv`

* comment_id: Unique identifier of the label comment
* title: Title of the submission
* comment : Label comment
* parent : Comment of interest (COI), can be a direct comment to submission 
* grandparent : a direct comment to the submission, parent of COI, can be empty
* child : child comment of the label comment
* fallacy_exists : If the fallacy exists or not
* fallacy_highlighted : The text marked as fallacious
* fallacy_highlighted_indices : The indices (span) of parent comment highlightes as fallacious
* claim : claim as described by the crowdworker

#### Metadata for `*_negative-samples.csv`

* comment_id: same as for `*_annotated.csv`
* title : same as for `*_annotated.csv`
* comment : same as for `*_annotated.csv`
* parent : same as for `*_annotated.csv`
* grandparent : same as for `*_annotated.csv`
* child : same as for `*_annotated.csv`
* fallacy_exists : always 0 (non-fallacious)
