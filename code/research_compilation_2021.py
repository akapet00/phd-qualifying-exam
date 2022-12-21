import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import update_matplotlib_rc_parameters


df = pd.read_excel(
    os.path.join(os.pardir, 'data', 'paper_count.ods')
    )
df_2021 = df[df['year']==2021].set_index('year').stack().reset_index()
df_2021 = df_2021.rename(columns={'level_1': 'study type', 0: 'count'})
df_2021.sort_values('count', ascending=False, inplace=True)

# visualize
update_matplotlib_rc_parameters()
fig, ax = plt.subplots()
sns.barplot(df_2021, x='study type', y='count', palette='rocket_r', ax=ax)
ax.set(
    xlabel='',
    ylabel='number of studies',
    yticks=[0, 125, 250, 375],
    yticklabels=[0, 125, 250, 375],
    )
# fig.savefig(os.path.join(
#     os.pardir, 'artwork', 'research_compilation_2021.pdf'
#     ), bbox_inches='tight')
plt.show()
