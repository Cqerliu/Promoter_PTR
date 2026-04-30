import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.gridspec import GridSpec
import matplotlib.projections as mpl_proj

# =====================Radar chart=====================
def radar_factory(num_vars, frame='polygon'):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
            else:
                raise ValueError("frame error")

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self, spine_type='circle', path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("frame error")

    try:
        mpl_proj.register_projection(RadarAxes)
    except:
        pass
    return theta

# ===================== data=====================
promoter_lengths = ['100bp','200bp','300bp','400bp','500bp','600bp']

# ===== only promoter =====
acc = [0.891, 0.905, 0.893, 0.892, 0.892, 0.892]
precision = [0.910, 0.912, 0.907, 0.906, 0.904, 0.918]
recall = [0.870, 0.899, 0.880, 0.875, 0.878, 0.863]
f1 = [0.889, 0.904, 0.892, 0.890, 0.890, 0.888]
auc = [0.939, 0.947, 0.939, 0.938, 0.936, 0.936]

# ===== fusion RNAP =====
acc_inc = [0.918, 0.915, 0.910, 0.907, 0.910, 0.913]
precision_inc = [0.927, 0.933, 0.908, 0.911, 0.919, 0.914]
recall_inc= [0.910, 0.898, 0.916, 0.904, 0.899, 0.913]
f1_inc = [0.917, 0.913, 0.911, 0.906, 0.909, 0.9153]
auc_inc = [0.964, 0.962, 0.962, 0.958, 0.961, 0.963]

# =====Calculate the improvement amplitude =====
acc_gain     = np.array(acc_inc) - np.array(acc)
prec_gain    = np.array(precision_inc) - np.array(precision)
recall_gain  = np.array(recall_inc) - np.array(recall)
f1_gain      = np.array(f1_inc) - np.array(f1)
auc_gain     = np.array(auc_inc) - np.array(auc)

# ===================== figure =====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

num_vars = 5
theta = radar_factory(num_vars, frame='polygon')
labels = ['ACC', 'Precision', 'Recall', 'F1', 'AUC']

color_list = ['#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#E53935', '#00ACC1']

fig = plt.figure(figsize=(24, 16))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)

axes = [
    fig.add_subplot(gs[0, 0], projection='radar'),
    fig.add_subplot(gs[0, 1], projection='radar'),
    fig.add_subplot(gs[0, 2], projection='radar'),
    fig.add_subplot(gs[1, 0], projection='radar'),
    fig.add_subplot(gs[1, 1], projection='radar'),
    fig.add_subplot(gs[1, 2], projection='radar'),
]

for i, ax in enumerate(axes):
    gain_data = [
        acc_gain[i],
        prec_gain[i],
        recall_gain[i],
        f1_gain[i],
        auc_gain[i]
    ]
    color = color_list[i]

    ax.plot(theta, gain_data, color=color, linewidth=6, marker='o', markersize=10)
    ax.fill(theta, gain_data, facecolor=color, alpha=0.4)

    for angle, label in zip(theta, labels):
        ax.text(angle, 0.081, label, 
                ha='center', va='center',
                fontsize=20, fontweight='bold')
    ax.set_xticklabels([]) 

    ax.set_ylim(-0.06, 0.06)

    # Rnage scale
    for tick in ax.get_yticklabels():
        tick.set_fontsize(15)
        tick.set_fontweight('bold')
        tick.set_color('#444444')

    ax.set_title(f'{promoter_lengths[i]}',
                 fontsize=26, fontweight='bold', y=-0.10)

    # Value display
    for j, v in enumerate(gain_data):
        text = f"{v:+.3f}"
        text_color = '#2E7D32' if v >= 0 else '#E53935'
        ax.text(theta[j], 0.045, text,
                ha='center', va='center',
                fontsize=24, fontweight='bold',
                color=text_color, zorder=10)

plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
plt.savefig('rn_gain_radar.png', dpi=300, bbox_inches='tight')
plt.show()