# 示范数据里，每个关节在第 2、3 步各变化多少度，以及第 3 步开始时的姿态（测水温和称重对照）
import json, glob, sys
import numpy as np
D = 180 / np.pi
for task in ("temp", "weigh"):
    d2, d3, p3, lag = [], [], [], []
    for f in glob.glob(f"{sys.argv[1]}/nero_{task}_v1/*.json"):
        t = json.load(open(f))
        q = np.asarray(t["proprios"]["arm_joint"]) * D
        s = np.asarray(t["source"]["step_index"])
        i2, i3 = np.where(s == 2)[0], np.where(s == 3)[0]
        if len(i2) < 2 or len(i3) < 2:
            continue
        d2.append(q[i3[0]] - q[i2[0]])
        d3.append(q[i3[-1]] - q[i3[0]])
        p3.append(q[i3[0]])
        lag.append(np.abs(q[i3] - q[i3[0]]).max(0))
    np.set_printoptions(precision=0, suppress=True)
    print(f"== {task}  示范 {len(d2)} 条（关节 1..7，单位度）")
    print("  第 2 步内变化 中位数", np.median(d2, 0))
    print("  第 3 步内变化 中位数", np.median(d3, 0))
    print("  第 3 步内最大偏离 中位数", np.median(lag, 0))
    print("  第 3 步开始时姿态 中位数", np.median(p3, 0))
    print("  第 3 步开始时姿态 p10   ", np.percentile(p3, 10, 0))
    print("  第 3 步开始时姿态 p90   ", np.percentile(p3, 90, 0))
