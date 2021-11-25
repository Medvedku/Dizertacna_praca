import pandas as pd
import math
concrete = {
    "Class": ["C12/15", 'C16/20', 'C20/25',
              'C25/30', 'C30/37', 'C35/45',
              'C40/50', 'C45/55', 'C50/60',
              'C55/67', 'C60/75', 'C70/85',
              'C80/95', 'C90/105']
}

df = pd.DataFrame(concrete)

def get_f_ctm(row):
    if row["f_ck"] <= 50:
        return 0.3 * row["f_ck"]**(2/3)
    else:
        return 2.12 * math.log(1+((row["f_cm"]/10)))
def get_eps_c1(row):
    if 0.7 * row["f_cm"]** 0.31 < 2.8:
        return 0.7 * row["f_cm"]** 0.31
    else:
        return 2.8
def get_eps_cu1(row):
    if row["f_ck"] <= 50:
        return 3.5
    else:
        return 2.8 + 27*((98-row["f_cm"])/100)**4
def get_eps_c2(row):
    if row["f_ck"] <= 50:
        return 2.0
    else:
        return 2.0 + 0.085*(row["f_ck"]-50)**0.53
def get_eps_cu2(row):
    if row["f_ck"] <= 50:
        return 3.5
    else:
        return 2.6 + 35*((90-row["f_ck"])/100)**4
def n(row):
    if row["f_ck"] <= 50:
        return 2.0
    else:
        return 1.4 + 23.4*((90-row["f_ck"])/100)**4
def get_eps_c3(row):
    if row["f_ck"] <= 50:
        return 1.75
    else:
        return 1.75 + 0.55*((row["f_ck"]-50)/40)
def get_eps_cu3(row):
    if row["f_ck"] <= 50:
        return 3.5
    else:
        return 2.6 + 35*((90-row["f_ck"])/100)**4

df["f_ck"] = df["Class"].apply(lambda x: x.split("/")[0].replace("C",""))
df["f_ck"] = df["f_ck"].apply(lambda x: int(x))
df["f_cube"] = df["Class"].apply(lambda x: x.split("/")[1].replace("C",""))
df["f_cube"] = df["f_cube"].apply(lambda x: int(x))
df["f_cm"] = df["f_ck"] + 8
df = df.assign(f_ctm = df.apply(get_f_ctm, axis=1))
df["f_ck_0.05"] = 0.7 * df["f_ctm"]
df["f_ck_0.905"] = 1.3 * df["f_ctm"]
df["E_cm"] = 22 * (df["f_cm"]/10)**(0.3)
df = df.assign(eps_c1 = df.apply(get_eps_c1, axis=1))
df = df.assign(eps_cu1 = df.apply(get_eps_cu1, axis=1))
df = df.assign(eps_c2 = df.apply(get_eps_c2, axis=1))
df = df.assign(eps_cu2 = df.apply(get_eps_cu2, axis=1))
df = df.assign(n = df.apply(n, axis=1))
df = df.assign(eps_c3 = df.apply(get_eps_c3, axis=1))
df = df.assign(eps_cu3 = df.apply(get_eps_cu3, axis=1))

concrete = "C30/37"

filt = (df["Class"]==concrete)

jozo = df.loc[filt, "E_cm"].values[0]

df.to_csv("Concrete_data.csv", index=False)
