# DLSCA 防禦端專案計畫書

**專案代號**：`dlsca-defense-v2`
**所屬專題**：基於生成對抗網路之主動式對抗旁通道防禦機制
**文件版本**：v0.1（規劃階段，尚未開始實作）
**與攻擊端的關係**：攻擊端 `dlsca-attack-v2` 是獨立 sibling repo，不共用程式碼、只共用資料格式介面（見 §2）。

---

## 1. 目標與核心論點

**核心論點**：用 GAN（或對抗訓練式擾動產生器）訓練出的防禦，在**同樣的擾動成本下**，比單純加高斯噪訊或時域干擾（desync-style jamming）能提供更好的防禦效率——用更少的成本，讓攻擊者需要更多軌跡才能破解金鑰。

這不是「GAN 防禦有沒有效」，是「GAN 防禦是否比簡單基準線**更有效率**」——實驗設計必須是**成本匹配的比較**，不是各自跑各自的、事後拿不同成本的結果比較。

## 2. 跟 dlsca-attack-v2 的介面

**不共用程式碼，只共用資料格式**——這是刻意的邊界，避免防禦端的實驗設計偷偷依賴攻擊端內部實作細節（也避免兩個 repo 的 git 歷史/版本相依糾纏在一起）。

防禦端要做的事：讀取已訓練好的攻擊者（`dlsca-attack-v2/runs/{run}/model.keras` + `config_snapshot.yaml` + `split_indices.npz`），輸出一個防禦後的 E 集軌跡陣列（`.npy`，shape 跟攻擊端 `attack_traces[split.e]` 完全一致），然後呼叫攻擊端既有的 Stage B 介面做評估：

```bash
# 防禦端只需要產出 defended_traces.npy，其餘全部交給攻擊端既有管線
python3 <dlsca-attack-v2>/scripts/02_run_attack.py \
  --run <dlsca-attack-v2>/runs/E01_baseline_clean_20260816_1302 \
  --traces defended_traces.npy --out results/probs.npy

python3 <dlsca-attack-v2>/scripts/03_evaluate.py \
  --run <dlsca-attack-v2>/runs/E01_baseline_clean_20260816_1302 \
  --probs results/probs.npy --out results/metrics.json
```

這保證防禦效果的判定（GE/N_TGE/PI，100次獨立重排攻擊）**跟攻擊端用的是同一套已經被 33 輪除錯驗證過的評估邏輯**，不會因為防禦端自己另外寫一套評估、寫出跟攻擊端不一致或不夠嚴謹的結果。

**成本指標**（`src/metrics/perturbation.py::psr/l2/linf`）是攻擊端已經實作好、防禦端直接複製過來用的純函式——不重新發明，見 §5。

## 3. 威脅模型（防禦端視角）

延續攻擊端 CLAUDE.md §5.1 的 profiled attack 威脅模型，防禦端這邊定義：

- **靜態攻擊者（A0，本階段目標）**：攻擊者已經訓練好一個固定模型（例如攻擊端的 E01），部署防禦前後都用同一個模型攻擊。這是最基本、最容易嚴謹評估的情境，也是本專案第一階段唯一目標。
- **自適應攻擊者（A1，後續延伸，不在本階段範圍）**：攻擊者發現防禦部署後重新訓練模型繞過。更有說服力但工程量大（需要防禦端先產出足量防禦後訓練資料，再讓攻擊端重跑一次完整訓練），留待第一階段結果出來後再決定是否投入。

**固定攻擊者的選擇是實驗設計的關鍵變數**（攻擊端 CLAUDE.md 附錄 C.2 已經證實：同一個防禦對「有沒有先天免疫力」的不同攻擊者效果差 4 倍以上）。本專案全程固定用 `dlsca-attack-v2` 的 **E01（clean baseline，未對任何防禦手法免疫過）**當基準攻擊者，不能換成 E02（噪訊增強訓練過，對高斯噪訊類防禦有內建免疫力，會系統性高估防禦效果）。

## 4. 舊版程式碼的已知問題（重新驗證前必須先修正）

`/mnt/c/Document/B11209017/ASCAD/GAN/train_improved_defender.py` 是上一輪專題已經做出來、有實測數據的對抗擾動產生器（非真GAN，無discriminator）。移植過來重新驗證前，先記錄兩個已知問題：

1. **訓練用的凍結攻擊者是 `cnnd_paper_model.h5`（`train_cnnd.py` 的產物）**——這正是攻擊端 CLAUDE.md 附錄 B.11 證實「幾乎沒學到東西」的那個模型（用嚴謹評估法重測，GE=199.27，比隨機基準還差）。**拿一個本身沒訓練起來的模型當對抗訓練的目標，防禦器學到的『讓攻擊者混淆』可能只是在利用這個模型的雜訊，不是真的學會了對抗一個有效攻擊者**。本專案的凍結攻擊者必須換成攻擊端已驗證有效的 E01（N_TGE=475）。
2. **舊版評估用的是攻擊端 `ASCAD_test_models.py::full_ranks()`**——單次、不重排、依原始儲存順序，正是攻擊端整個重構專案存在的理由（附錄 B.11 的「N_TGE≈100 是量測假象」那個坑）。舊版 `defender_summary.csv` 的「>1000條軌跡才破解」這個數字，可信度跟當年的「N_TGE≈100」是同一個等級，**不能直接引用，必須用攻擊端現在的100-run評估法重新量測**。

## 5. 專案結構

```
dlsca-defense-v2/
├── configs/
│   ├── base.yaml
│   ├── attacker/               # 指向 dlsca-attack-v2 已訓練攻擊者的路徑設定
│   │   └── e01_baseline.yaml   # attacker_run: <path-to-dlsca-attack-v2>/runs/E01_baseline_clean_...
│   └── exp/                    # D01, D02... 防禦實驗編號（比照攻擊端E01-E08慣例）
├── src/
│   ├── config.py                # 沿用攻擊端同一套 YAML 合併邏輯（複製，不 import）
│   ├── generator/
│   │   └── conv_perturber.py    # 移植 build_improved_defender()，見 §6
│   ├── train/
│   │   └── adversarial.py       # 對抗訓練迴圈，凍結攻擊者、只更新產生器
│   └── bridge/
│       └── attack_interface.py  # 讀 dlsca-attack-v2 的 model.keras/config_snapshot.yaml/split_indices.npz，
│                                 # 封裝成「餵 raw trace 進去、拿到 softmax 機率」的函式，供訓練迴圈算對抗loss用
├── scripts/
│   ├── 01_train_defender.py     # 對抗訓練，輸出 generator.keras
│   ├── 02_generate_defended.py  # generator 對 dlsca-attack-v2 的 E 集跑一次，輸出 defended_traces.npy
│   │                             # （之後接 dlsca-attack-v2 的 02/03，見 §2）
│   └── 03_compare_defenses.py   # 讀多個 metrics.json + cost_metrics.json，畫 PSR-vs-GE 比較圖
├── src/metrics/
│   └── perturbation.py          # 從 dlsca-attack-v2 複製 psr/l2/linf（同介面，同測試），保持成本指標一致
├── runs/                        # 訓練產物，git-ignored
├── results/                     # 防禦評估產物（呼叫攻擊端Stage B後的probs.npy/metrics.json落地處），git-ignored
├── tests/
├── docs/
│   └── runs.md                  # 比照攻擊端慣例
├── requirements.txt
└── README.md
```

## 6. 產生器架構（第一版，移植自舊專題）

延續 §4 的修正，架構本身先原封不動移植（已知能訓練，問題出在訓練目標不是產生器架構）：

```
Input(700,1)
  → Conv1D(32,k7,relu) → BN
  → Conv1D(32,k7,relu) → BN → Add(殘差)
  → Conv1D(16,k5,relu) → BN
  → Conv1D(8,k5,relu)  → BN
  → Conv1D(1,k3,tanh)
  → Lambda(x * epsilon)                    # 有界擾動 [-epsilon, epsilon]
```

**輸入輸出的前處理紀律**（新增，舊版沒處理）：產生器對**原始 raw trace**（跟攻擊端 `attack_traces`/`profiling_traces` 同一個 int8 尺度，cast float32）輸出擾動，`defended = raw + perturbation`，這個 raw 尺度的 `defended` 就是要存成 `.npy` 交給攻擊端 `02_run_attack.py --traces` 的東西——攻擊端會自己用它既有的 Standardizer/MinMax 流程把它轉換成攻擊模型吃的尺度，防禦端不用管這一段，也不該自己另外標準化（那樣輸出的 `.npy` 尺度會跟攻擊端期待的 raw 尺度對不上）。

**對抗訓練時**：為了算 `attacker(x_defended)` 的 loss，防禦端的訓練迴圈裡需要複製一份跟攻擊端一致的 Standardizer/MinMax（`src/bridge/attack_interface.py` 負責讀取攻擊端 `config_snapshot.yaml` 裡的 `preprocess` 設定、在 A 集上重新 fit，邏輯跟攻擊端 `02_run_attack.py` 重新 fit 的方式一致）——訓練時的 forward pass 是 `raw → +perturbation → standardize → minmax → frozen_attacker → loss`，跟部署/評估時最終會發生的事情完全一致。

**Loss（第一版，原封不動移植)**：`confusion(→均勻分佈的交叉熵) + λ_L2·||perturbation||² + λ_smooth·相鄰點平方差 + λ_entropy·|熵-log(256)|`，四個權重維持舊版的量級當起點（`λ_confuse=1.0, λ_L2=0.005, λ_smooth=0.002, λ_entropy=0.1`），重新驗證後再決定要不要調。

## 7. 成本指標與比較基準

`src/metrics/perturbation.py` 複製攻擊端同一份實作（`psr`/`l2`/`linf`，逐軌跡陣列輸出，不是純量）。跟攻擊端「高斯噪訊 PSR-vs-N_TGE」基準曲線（附錄 C.3，8個sigma_ratio點，PSR 0.0089–0.2674）比較時，**兩邊的 PSR 都要用同一個定義、對同一批 clean E 集算**，才能公平疊在同一張圖上。

## 8. 實驗編號規劃（草案）

| 編號 | 內容 | 依賴 |
|---|---|---|
| D01 | 對抗擾動產生器 vs E01（凍結攻擊者），重新驗證舊專題的假設 | §4/§6 修正過的版本 |
| D02 | D01 的 epsilon/loss權重掃描，找出效率最好的配方 | D01 跑通 |
| D03 | 跟高斯噪訊基準曲線做 PSR 成本匹配比較 | D01/D02 + 攻擊端附錄C.3 |
| D04 | 時域干擾基準線（desync-style jamming，用攻擊端已有的 ASCAD_desync50/100 概念另外建一版） | 獨立於D01-D03，可平行做 |
| D05（延伸，未定案） | 自適應攻擊者（A1）情境：防禦部署後攻擊端重訓練 | D01-D03 有結果後再決定 |

## 9. 驗收標準（草案）

- [ ] D01 用嚴謹100-run評估法重新驗證舊專題假設，得出誠實數字（不管是否比舊版好）
- [ ] 至少一組 GAN 防禦配方，在**同樣PSR成本**下，GE@9000/N_TGE 比高斯噪訊基準曲線同一個PSR點更好
- [ ] 三方比較圖（噪訊 vs 干擾 vs GAN）可以直接放進期末報告
- [ ] 防禦訓練/評估流程可重現（config快照、種子、env.json，比照攻擊端紀律）

## 10. 已知風險

| 風險 | 對策 |
|---|---|
| 舊版凍結攻擊者不可靠（§4） | 換成攻擊端 E01，第一步就做 |
| 防禦端自己另外寫評估邏輯，跟攻擊端不一致 | 強制走 Stage B 介面，不自己算 GE |
| PSR 定義兩邊沒對齊 | 直接複製攻擊端 `perturbation.py`，不重新實作 |
| 靜態攻擊者情境下的「勝利」對自適應攻擊者沒說服力 | 明確標註 D01-D04 是A0情境，A1留作後續延伸，不要混著講 |
