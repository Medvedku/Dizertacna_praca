import pandas as pd
import math
rebar = {
    "Class": ["B420B", 'B500B']
}
df = pd.DataFrame(rebar)
df["f_yk"] = df["Class"].apply(lambda x: x.replace("B",""))
df["f_yk"] = df["f_yk"].apply(lambda x: int(x))
df["E_s"]  = 200*1e9
df.to_csv("Rebar_data.csv", index=False)
