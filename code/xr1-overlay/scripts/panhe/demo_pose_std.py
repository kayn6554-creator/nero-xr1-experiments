# 示范数据里，每个关节在第 2、3 步各变化多少度，以及第 3 步开始时的姿态（只用标准库）
import json, glob, sys, math, statistics as st
D = 180 / math.pi
def pct(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p / 100; f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)
def show(name, rows, fn=st.median):
    print(f"  {name:12s}", " ".join(f"{fn([r[j] for r in rows]):7.1f}" for j in range(7)))
for task in ("temp", "weigh"):
    d2, d3, p3, lag = [], [], [], []
    for f in glob.glob(f"{sys.argv[1]}/nero_{task}_v1/*.json"):
        t = json.load(open(f))
        q = [[v * D for v in row] for row in t["proprios"]["arm_joint"]]
        s = t["source"]["step_index"]
        i2 = [i for i, v in enumerate(s) if v == 2]; i3 = [i for i, v in enumerate(s) if v == 3]
        if len(i2) < 2 or len(i3) < 2:
            continue
        a, b, c = q[i2[0]], q[i3[0]], q[i3[-1]]
        d2.append([b[j] - a[j] for j in range(7)])
        d3.append([c[j] - b[j] for j in range(7)])
        p3.append(b)
        lag.append([max(abs(q[i][j] - b[j]) for i in i3) for j in range(7)])
    print(f"== {task}  示范 {len(d2)} 条（关节 1..7，单位度，中位数）")
    print("  " + " " * 12, " ".join(f"{'j'+str(j+1):>7s}" for j in range(7)))
    show("第2步内变化", d2); show("第3步内变化", d3); show("第3步最大偏离", lag)
    show("第3步起姿态", p3)
    show("  p10", p3, lambda x: pct(x, 10)); show("  p90", p3, lambda x: pct(x, 90))
