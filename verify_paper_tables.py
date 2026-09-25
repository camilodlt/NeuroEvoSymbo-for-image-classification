#!/usr/bin/env python3
"""Check every results table in the paper against the result files shipped in this folder.

Run from anywhere:

    python3 verify_paper_tables.py

It parses the LaTeX sources in the local manuscript directory and compares every numeric cell of
Table 1 and Tables 4, 7 to 17 against the csv, json and txt files under the dataset folders
here. Exit status is 0 when every cell agrees to one decimal place.

Sources, all relative to this file's directory:

  MAGE end-to-end   <ds>/MAGE_ALONE_20H/mage_imgcls/metrics_per_individual_<split>_alltrue.csv
                    The validation-selected run is the one with the highest validation BACC.
                    Means use the sample standard deviation over the 16 runs.
  Distillation      <ds>/<trial>/ml_models_boost_0/extended_metrics_<metric>_<dict>.csv, test rows.
                    <dict> is beststrue_false, bestsfalse_false or bestsfalse_true for the
                    Best, Runs and All per dimension dictionaries. The ECE rows read the
                    bacc-scored file, because scikit-learn has no ECE scorer and the ECE runs
                    select hyperparameters with balanced accuracy.
  Neural baselines  <ds>/NN_baselines/<backbone>_<name>/extended_metrics_best_val_bacc.json
                    and per_class_test.csv for the per-class tables.
  Teacher           <ds>/<trial>/nn_surrogates_boost_*/metrics.txt, three values, the third
                    being test BACC.
"""
import csv, json, pathlib, re, statistics as st, sys

HERE  = pathlib.Path(__file__).resolve().parent
PAPER = next(p for p in HERE.iterdir() if p.is_dir() and p.name.endswith("_Paper"))
TOL   = 0.051          # one decimal place, with round-half-up at the boundary

DS = {  # paper name -> (dataset folder, trial id, NN folder suffix, n classes)
 "BloodMNIST" : ("BloodMNIST",  "L16_WD01_asinh_R18_128", "BloodMNIST",   8),
 "OrganAMNIST": ("ORGANAMNIST", "L16_WD01_asinh_R18_128", "OrganAMNIST", 11),
 "OrganCMNIST": ("ORGANCMNIST", "L16_WD01_asinh_R18_128", "OrganCMNIST", 11),
 "OCTMNIST"   : ("OCTMNIST",    "L16_WD01_asinh_R18_512", "OCTMNIST",     4),
 "PathMNIST"  : ("PathMNIST",   "L16_WD01_asinh_R18_512", "PathMNIST",    9),
}
T1NAME   = {"Blood":"BloodMNIST","Organ A":"OrganAMNIST","Organ C":"OrganCMNIST",
            "OCT":"OCTMNIST","Path":"PathMNIST"}
DICTS    = ["beststrue_false","bestsfalse_false","bestsfalse_true"]
BB       = ["resnet18","resnet34","resnext50"]
SRC      = {"bacc":("bacc","ece"),"macro_f1":("macro_f1",),"ece":("bacc","ece"),"auroc":("auroc",)}
ROW      = {r"\Gls{bacc} (\%)":"bacc", r"Macro F1 (\%)":"macro_f1",
            r"ECE (\%)":"ece", r"AUROC (\%)":"auroc"}

problems, checked = [], 0

def note(where, what, paper, truth):
    global checked
    if paper is None: return
    checked += 1
    if abs(paper - round(truth, 1)) > TOL:
        problems.append((where, what, paper, round(truth, 2)))

def num(s):
    m = re.search(r"-?[0-9]+\.?[0-9]*", s.replace(r"\textbf{","").replace("}",""))
    return float(m.group()) if m else None

def e2e(ds):
    base = HERE / DS[ds][0] / "MAGE_ALONE_20H/mage_imgcls"
    val = list(csv.DictReader((base/"metrics_per_individual_val_alltrue.csv").open()))
    tst = list(csv.DictReader((base/"metrics_per_individual_test_alltrue.csv").open()))
    sel = max(val, key=lambda r: float(r["bacc"]))["individual"]
    tmap = {r["individual"]: r for r in tst}
    d = {}
    for m in ("bacc","macro_f1","ece","auroc"):
        v = [100*float(r[m]) for r in tst]
        d[m] = {"sel":100*float(tmap[sel][m]), "mean":st.mean(v), "sd":st.stdev(v)}
    return d

def distill(ds, metric, dic):
    out, trial = DS[ds][0], DS[ds][1]
    for cand in SRC[metric]:
        p = HERE/out/trial/"ml_models_boost_0"/f"extended_metrics_{cand}_{dic}.csv"
        if p.exists():
            r = {x["model"]: x for x in csv.DictReader(p.open()) if x["split"]=="test"}
            if {"RF","SVC","LR"} <= set(r): return r
    return None

def nn(ds, metric):
    out, suf = DS[ds][0], DS[ds][2]
    d = {}
    for b in BB:
        j = HERE/out/"NN_baselines"/f"{b}_{suf}"/"extended_metrics_best_val_bacc.json"
        d[b] = 100*json.load(j.open())["test"][metric] if j.exists() else None
    return d

TRUTH = {ds: e2e(ds) for ds in DS}
body  = (PAPER/"body.tex").read_text()
app   = (PAPER/"appendix.tex").read_text()

# ---- Table 1 ----
for line in body.split("\n"):
    m = re.match(r"^(Blood|Organ A|Organ C|OCT|Path)\s+&", line)
    if not m: continue
    ds = T1NAME[m.group(1)]; c = [x.strip() for x in line.split("&")]; t = TRUTH[ds]["bacc"]
    note(f"Table 1 {ds}","E2E best on val", num(c[2]), t["sel"])
    note(f"Table 1 {ds}","E2E avg",         num(c[3]), t["mean"])
    note(f"Table 1 {ds}","E2E std",         num(c[3].split("pm")[1]), t["sd"])
    cells = [x.strip() for x in c[4:13]]
    for dic, g in zip(DICTS, [cells[0:3], cells[3:6], cells[6:9]]):
        rows = distill(ds, "bacc", dic)
        for head, key, pv in zip(("RF","SVM","LR"), ("RF","SVC","LR"), g):
            note(f"Table 1 {ds}", f"{dic} {head}", num(pv), 100*float(rows[key]["bacc"]))
    for b, pv in zip(BB, c[13:16]):
        note(f"Table 1 {ds}", f"NN {b}", num(pv), nn(ds,"bacc")[b])
    note(f"Table 1 {ds}", "delta", num(c[16]),
         max(num(x) for x in c[13:16]) - max(num(x) for x in cells))

# ---- Table 4, teacher against the baselines ----
i = app.find("of the bottlenecked teacher against the unconstrained")
if i >= 0:
    blk = app[i: app.index(r"\end{table}", i)]
    for line in blk.split("\n"):
        for ds in DS:
            if not line.strip().startswith(ds): continue
            c = [x.strip() for x in line.rstrip().rstrip("\\").split("&")]
            out, trial = DS[ds][0], DS[ds][1]
            f = next((HERE/out/trial).glob("nn_surrogates_boost_*/metrics.txt"), None)
            if f: note(f"Table 4 {ds}","teacher", num(c[1]),
                       100*float(f.read_text().strip().split(",")[2]))
            for b, pv in zip(BB, c[2:5]):
                note(f"Table 4 {ds}", f"NN {b}", num(pv), nn(ds,"bacc")[b])

# ---- Tables 8 to 12, per-dataset metric tables ----
for ds in DS:
    cap = f"Test performance on the {ds} dataset."
    blk = app[app.index(cap):]; blk = blk[:blk.index(r"\end{table}")]
    for lab, met in ROW.items():
        seg = blk[blk.index(lab):]; seg = seg[:seg.index(r"\\")]
        cells = [x.strip() for x in seg.split("&")][1:]
        t = TRUTH[ds][met]
        note(f"{ds} {met}","E2E sel",  num(cells[0]), t["sel"])
        note(f"{ds} {met}","E2E avg",  num(cells[1]), t["mean"])
        note(f"{ds} {met}","E2E std",
             num(cells[1].split("pm")[1]) if "pm" in cells[1] else None, t["sd"])
        for dic, g in zip(DICTS, [cells[2:5], cells[5:8], cells[8:11]]):
            rows = distill(ds, met, dic)
            for head, key, pv in zip(("RF","SVM","LR"), ("RF","SVC","LR"), g):
                note(f"{ds} {met}", f"{dic} {head}", num(pv), 100*float(rows[key][met]))
        for b, pv in zip(BB, cells[11:14]):
            note(f"{ds} {met}", f"NN {b}", num(pv), nn(ds,met)[b])
        if len(cells) > 14 and num(cells[14]) is not None:
            dist = [num(x) for x in cells[2:11] if num(x) is not None]
            nnl  = [num(x) for x in cells[11:14]]
            note(f"{ds} {met}", "delta", num(cells[14]),
                 abs(min(nnl)-min(dist)) if met=="ece" else max(nnl)-max(dist))

# ---- Table 7, cross-dataset summary ----
cap = "Summary of additional test metrics across datasets."
blk = app[app.index(cap):]; blk = blk[:blk.index(r"\end{table}")]
lines = blk.split("\n")
for i, l in enumerate(lines):
    if l.strip() not in T1NAME: continue
    ds = T1NAME[l.strip()]
    rows3 = [[x.strip() for x in lines[j].split("&")[1:]] for j in range(i+1, i+4)]
    for met, row in zip(("macro_f1","ece","auroc"), rows3):
        agg = min if met == "ece" else max
        note(f"Table 7 {ds} {met}","E2E", num(row[0]), TRUTH[ds][met]["sel"])
        for dic, pv in zip(("beststrue_false","bestsfalse_false"), row[1:3]):
            rows = distill(ds, met, dic)
            if rows is None or num(pv) is None: continue
            note(f"Table 7 {ds} {met}", f"{dic} best head", num(pv),
                 agg(100*float(rows[k][met]) for k in ("RF","SVC","LR")))
        good = [v for v in nn(ds,met).values() if v is not None]
        note(f"Table 7 {ds} {met}","NN best", num(row[3]), agg(good))

# ---- Tables 13 to 17, per-class tables ----
PC = PAPER/"appendix/per_class"
SHORT = {"BloodMNIST":"blood","OrganAMNIST":"organa","OrganCMNIST":"organc",
         "OCTMNIST":"oct","PathMNIST":"path"}
def colvec(path, col):
    return [100*float(r[col]) for r in csv.DictReader(path.open())]
for ds,(out,trial,suf,nc) in DS.items():
    tex = (PC/f"{SHORT[ds]}.tex").read_text().split("\n")
    rows = [l for l in tex if l.rstrip().endswith(r"\\") and l.count("&") >= 10
            and "multicolumn" not in l and "end-to-end" not in l]
    e2ep = HERE/out/"MAGE_ALONE_20H/mage_imgcls/per_class_test_allfalse.csv"
    truth = [("MAGE end-to-end", colvec(e2ep,"sensitivity"), colvec(e2ep,"specificity"))]
    for dic in DICTS:
        f = HERE/out/trial/"ml_models_boost_0"/f"extended_metrics_macro_f1_{dic}.csv"
        d = {r["model"]: r for r in csv.DictReader(f.open()) if r["split"]=="test"}
        for key in ("RF","SVC","LR"):
            truth.append((f"{dic} {key}",
                          [100*float(x) for x in d[key]["sensitivity"].split(";")],
                          [100*float(x) for x in d[key]["specificity"].split(";")]))
    for b in BB:
        p = HERE/out/"NN_baselines"/f"{b}_{suf}"/"per_class_test.csv"
        truth.append((b, colvec(p,"sensitivity"), colvec(p,"specificity")))
    for ci, line in enumerate(rows):
        cells = [c.strip() for c in line.rstrip().rstrip("\\").split("&")][1:]
        for (colname, sens, spec), cell in zip(truth, cells):
            m = re.findall(r"(?:\\textbf\{)?(-{2}|[0-9]+\.[0-9])\}?", cell)
            if len(m) != 2 or "--" in m:
                problems.append((f"per-class {ds}", f"class {ci} {colname}", cell, "pending"))
                continue
            note(f"per-class {ds}", f"class {ci} {colname} sens", float(m[0]), sens[ci])
            note(f"per-class {ds}", f"class {ci} {colname} spec", float(m[1]), spec[ci])

print(f"checked {checked} values against the result files in {HERE.name}/")
if problems:
    print(f"\n{len(problems)} disagreement(s):")
    for p in problems:
        print(f"   {p[0]:<24}{p[1]:<34}paper={p[2]:>7}  file={p[3]:>7}")
    print("\nNote: a 0.05 difference is round-half-up at the boundary and is not an error.")
else:
    print("every cell agrees")
sys.exit(1 if any(abs(float(p[2])-float(p[3])) > 0.0501 for p in problems
                  if isinstance(p[2],(int,float)) and isinstance(p[3],(int,float))) else 0)
