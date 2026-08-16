# 執行紀錄

索引 `runs/`（git-ignored）底下的實際執行結果。詳細分析見 `CLAUDE.md` 附錄 A。

| run_dir | 攻擊者 | n_train | epochs（實際/上限） | PSR mean | GE@9000 | N_TGE | PI |
|---|---|---|---|---|---|---|---|
| `D01_replicate_baseline_20260816_232552_948723` | E01（GPU server） | 15000 | 34/40（early stop） | 0.0249 | 111.84 | None | -0.0945 |
