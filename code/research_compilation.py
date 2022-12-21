import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import update_matplotlib_rc_parameters


df = pd.read_excel(
    os.path.join(os.pardir, 'data', 'paper_count.ods')
    )

# visualize
update_matplotlib_rc_parameters()
cs = sns.color_palette('rocket', 4)
fig, ax = plt.subplots()
ax.plot(df['year'], df['epidemiological'], 'o-', markevery=10, c=cs[0], lw=3,
        label='epidemiological')
ax.plot(df['year'], df['review'], '^--', markevery=10, c=cs[1], lw=3,
        label='review')
ax.plot(df['year'], df['experimental'], 's-.', markevery=10, c=cs[2], lw=3,
        label='experimental')
ax.plot(df['year'], df['dosimetric'], 'd:', markevery=10, c=cs[3], lw=3,
        label='dosimetric')
ax.set(
    xlabel='year',
    ylabel='number of studies',
    xticks=[1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020],
    xticklabels=[1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020],
    yticks=[0, 125, 250, 375],
    yticklabels=[0, 125, 250, 375],
    )
ax.legend(loc='best', title='studies', frameon=False)
# fig.savefig(os.path.join(
#     os.pardir, 'artwork', 'research_compilation.pdf'
#     ), bbox_inches='tight')
plt.show()
