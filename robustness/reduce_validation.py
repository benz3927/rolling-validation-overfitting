"""Shrink robustness/validation_n3000.json (about 150 MB) to a gzipped copy under GitHub's
file-size limit. Keeps candidates opt/over40/over50 and terms rv, signal, B, C."""
import gzip
import json

import numpy as np

R = json.load(open("robustness/validation_n3000.json"))
rec = np.array(R["rec"])
names, terms = R["candidate_names"], R["settings"]["terms"]
keep_c, keep_t = ["opt", "over40", "over50"], ["rv", "signal", "B", "C"]
sub = rec[:, :, [names.index(c) for c in keep_c]][..., [terms.index(t) for t in keep_t]]
R["rec"] = np.round(sub, 10).tolist()
R["candidate_names"] = keep_c
R["settings"]["terms"] = keep_t
R["settings"]["note"] = "reduced copy: candidates opt/over40/over50 and terms rv, signal, B, C only"
with gzip.open("robustness/validation_n3000.json.gz", "wt") as f:
    json.dump(R, f)
print("wrote robustness/validation_n3000.json.gz")
