import pandas as pd
df = pd.DataFrame([[1,2], [3,4]])
s = df.style.highlight_max(axis=None,
                           props='cellcolor:{red}; bfseries: ;')
s.to_latex()