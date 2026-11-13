# 补丁

`113-serving-from-atlas-20261113.patch`：用户上传的 Atlas 服务文件到 113 实际服务文件的差异，含 EF IK、逐请求日志和提速。适用起点不是原始上游，而是已经包含 M1/M2 的 2026-11-13 Atlas 快照。

先 `git apply --check` 再应用。不得再叠加同效果的 apply_ef_serve / apply_ef_log / apply_ef_fast。依赖与完整限制见 [代码说明](../code/README.md)。此补丁不包含 M5 或 M4/EF4 服务支持。
