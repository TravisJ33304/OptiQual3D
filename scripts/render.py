from optiqual3d.config.settings import VisualizationConfig

import numpy as np, glob, random
from optiqual3d.visualization.renderer import render_point_cloud

# Pick a random anomalous sample
files = glob.glob("datasets/generated/anomalous/*.npz")
d = np.load(random.choice(files))
points, mask = d["points"], d["mask"].astype(float)

render_point_cloud(points, scores=mask, title="Sample",
                   cfg=VisualizationConfig(backend="plotly"),
                   save_path="outputs/sample.html")