import subprocess, os, math, random, json
from concurrent.futures import ThreadPoolExecutor
BIN = "/tmp/claude-0/-home-claude/b1bc4455-e770-57c6-ada9-91dcfd104e34/scratchpad/AC5/myfilter"
def run_cells(cells, T, burn, batches=100, workers=2, chunk=40):
    """cells: list of (model,err,proxy,p,alpha,seed) -> list of (mean,se,bad)"""
    chunks = [cells[i:i+chunk] for i in range(0, len(cells), chunk)]
    def one(c):
        inp = "".join(f"{m} {e} {pr} {p!r} {a!r} {sd}\n" for m,e,pr,p,a,sd in c)
        out = subprocess.run([BIN,str(T),str(burn),str(batches)],input=inp,capture_output=True,text=True,check=True).stdout.split("\n")
        res=[]
        for l in out[:len(c)]:
            mm,se,bad=l.split(); res.append((float(mm),float(se),int(bad)))
        return res
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(one, chunks))
    return [r for part in parts for r in part]
def emit(results):
    path=os.environ.get("CLAIMSET_RESULTS")
    if path: json.dump(results, open(path,"w"))
def wrong_dir(model, err, p, mean, se, bad, z=4):
    """True if the bias is on the wrong side by more than z SE (or state non-finite)"""
    if bad: return True
    b = mean - p
    if err == "M":
        return (b > z*se) if model=="i" else (b < -z*se)
    # additive: expected: towards 0.5 and not beyond 0.5
    if p < 0.5: return (b < -z*se) or (mean > 0.5 + z*se)
    if p > 0.5: return (b > z*se) or (mean < 0.5 - z*se)
    return abs(b) > z*se
