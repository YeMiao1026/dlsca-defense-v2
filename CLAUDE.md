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
| D02 | D01 的 loss權重掃描，找出效率最好的配方 | D01 跑通 |
| D03 | **改編號**——loss權重掃描（D02）推翻了「smooth/l2拖累效率」假設後，改測「確定性擾動 vs 高斯噪訊獨立隨機性」這個結構性假設，見附錄A.5/D03 | D02 跑通 |
| D04 | **改編號**——SNR診斷（附錄A.8）找出真正機制（洩漏點覆蓋不完整）後，重新設計loss加入可微分洩漏抑制項，見附錄A.9/D04 | D01-D03 + SNR診斷完成 |
| D05（原D04/D03） | 跟高斯噪訊基準曲線做 PSR 成本匹配比較（三方比較圖） | D01-D04 找到至少一個能打平/贏過高斯基準線的配方後再進行 |
| D06（原D05/D04） | 時域干擾基準線（desync-style jamming，用攻擊端已有的 ASCAD_desync50/100 概念另外建一版） | 獨立於D01-D05，可平行做 |
| D07（原D06/D05，延伸，未定案） | 自適應攻擊者（A1）情境：防禦部署後攻擊端重訓練 | D01-D05 有結果後再決定 |

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

---

## 附錄 A：D01 執行結果

### A.1 產生器/攻擊者橋接（§4/§6 移植）已驗證

`src/bridge/attack_interface.py` 讀取 dlsca-attack-v2 的 `config_snapshot.yaml`＋`split_indices.npz`，在同一個 A 集上重新 fit Standardizer/MinMaxScaler，並把 transform 重新表達成 TF 常數運算，讓對抗訓練迴圈可以對「凍結攻擊者的 loss 對擾動」求梯度。`tests/test_bridge.py` 用合成資料驗證這個 TF 版本跟攻擊端 numpy 版本數值一致（這正是攻擊端 CLAUDE.md B.17/B.19/B.33 反覆踩過的「表面上做對了、實際上沒對齊」那類 bug 的預防性測試）。`BoundedPerturbation` 取代舊版的 `Lambda`，讓 `generator.keras` 可以正常 `save`/`load_model` 而不需要 `enable_unsafe_deserialization()`。

小規模（200條軌跡、2 epoch）在真實 E01 攻擊者上跑過整條鏈路（訓練→產生防禦波形→餵進攻擊端 `02_run_attack.py`/`03_evaluate.py`），確認接線正確、`metrics.json` 正常產出。

### A.2 D01 正式跑（GPU server，15000條軌跡，架構/loss權重原封不動沿用舊版）

`runs/D01_replicate_baseline_20260816_232552_948723/`。訓練在 epoch 34 觸發 patience=5 早停（loss 從 epoch 1 的 7.62 降到穩定在 5.605-5.607，confuse 項貼著 log(256)=5.545 的均勻分佈下限附近但沒有真正壓到底，l2/smooth 兩項持續下降代表擾動幅度在收斂過程中越變越小）。

對 E01 的 E 集（10000條軌跡）套用訓練好的產生器，成本：

```
PSR: mean=0.0249  median=0.0249  p90=0.0255  max=0.0265（各軌跡幾乎一致，變異極小）
L2:  mean=18.58
Linf: mean=2.38（遠低於 epsilon=6.0 上限，代表產生器並沒有把擾動預算用滿）
```

用攻擊端 Stage B 介面（`02_run_attack.py --traces/--out` + `03_evaluate.py`，100次獨立重排、`max_traces=9000`，跟附錄 C.3 高斯基準曲線同一套評估法）：

```
N_TGE  = None（未收斂）
N_SR90 = None
GE @ N=9000  = 111.84（100次獨立攻擊，全程在110-123之間緩慢震盪，沒有隨N拉寬持續下降的收斂跡象）
SR1 @ N=9000 = 0.0000
PI           = -0.0945（輕微負值，不是"confidently wrong"的崩潰量級）
```

### A.3 誠實結論：這一版跟同成本的高斯噪訊基準線比，效率明顯更差

對照攻擊端附錄 C.3 的高斯噪訊 PSR 掃描（同樣固定 E01 為攻擊者、同一套 100-run 評估法）：

| 防禦手法 | PSR | N_TGE | GE@9000 |
|---|---|---|---|
| 高斯噪訊 sigma_ratio=0.1 | 0.0089 | 514 | 0.00（完全收斂） |
| 高斯噪訊 sigma_ratio=0.25 | 0.0223 | 775 | 0.00（完全收斂） |
| **D01 GAN 產生器（本次）** | **0.0249** | **None** | **111.84（幾乎沒有真正壓垮攻擊者）** |
| 高斯噪訊 sigma_ratio=0.5 | 0.0446 | 2276 | 0.00（完全收斂） |

**在幾乎相同（甚至略高）的 PSR 成本下，單純的高斯噪訊讓攻擊者完全瓦解（GE精確降到0、100次攻擊全部收斂），這次移植的 GAN 產生器卻只把 GE 從隨機基準127.5壓到111.84，離攻擊者真正被瓦解還差得遠。** 這跟這個專案最初想證明的方向（GAN 防禦應該比噪訊更有效率）正好相反——這是舊版 `train_improved_defender.py` 架構與 loss 權重原封不動移植後，換上真正有效的攻擊者、換上嚴謹評估法之後得到的誠實結果，不是移植過程有 bug（§4 提到的兩個已知問題都已修正，且整條鏈路有測試與端到端smoke test驗證）。

**初步機制解讀（尚未驗證，留待後續投入判斷是否深究）**：高斯噪訊是逐點獨立同分佈的隨機噪訊，直接推高每個時間點的類內變異數，等同直接打壓 SNR 的分母；D01 訓練出的擾動 PSR 雖然數值相近，但由 `train_history.csv` 看得出來 l2/smooth 兩項在訓練過程中持續被壓低（代表 loss 權重把產生器推向「幅度小、相鄰點平滑」的解），而且 PSR 在10000條軌跡間幾乎不變（變異度極小，見上表 mean≈median≈p90）——這意味著產生器學出來的擾動對所有軌跡而言可能高度相似（低軌跡間變異），比較像是對洩漏點加了一個接近固定的偏移量，而不是像高斯噪訊那樣對每條軌跡注入獨立隨機性。一個固定偏移量會被 Standardizer/MinMaxScaler（在乾淨 A 集上 fit、對所有 E 軌跡套用同一組逐點統計量）部分「內部消化」掉——如果攻擊者本來就是靠逐點統計量在做判別，一個系統性但軌跡間一致的偏移，對「拉開類間變異、增加類內變異」這個判別任務的破壓效果，天生就會比同成本的隨機噪訊弱。**這是否為真正機制、以及loss權重（尤其是lambda_smooth）是否是主因，需要 D02 的超參數掃描才能確認，這裡先誠實記錄現象與初步假設，不下定論。**

### A.4 D02：loss 權重掃描，四點全部驗證 A.3 假設不成立

四個設定，架構/epsilon/confuse/entropy權重與n_train/epochs上限全部沿用D01，只動`lambda_l2`/`lambda_smooth`（皆用攻擊端Stage B介面、100-run、max_traces=9000評估，跟D01/附錄C.3同一套方法）：

| 設定 | lambda_l2 | lambda_smooth | PSR mean | GE@9000 | PI |
|---|---|---|---|---|---|
| **D01（基準）** | 0.005 | 0.002 | **0.0249** | **111.84** | -0.0945 |
| D02_nosmooth | 0.005 | 0.0 | 0.0393 | 137.32 | -0.0959 |
| D02_lowreg | 0.0005 | 0.0002 | 0.0636 | 194.83 | -0.0974 |
| D02_nol2 | 0.0 | 0.002 | 0.0974 | 187.70 | -0.0969 |
| D02_noreg | 0.0 | 0.0 | 0.1493 | 197.76 | -0.0974 |

**四點結果完全一致、方向跟A.3的假設相反**：拿掉/降低正則化後，PSR成本全部上升（最多到D01的6倍），但GE不但沒有變好（更低），反而全部變差（更高，多數還輸給隨機基準127.5）——D01自己的（滿正則化）配方，是這五個裡面唯一一個把GE壓到隨機基準以下的。換句話說：**smooth/l2懲罰不是在拖累防禦效果，拿掉它們讓訓練變得明顯更差**，即使花更多成本也一樣。A.3提出的「D01效率不如高斯噪訊是因為smooth/l2把擾動壓成太平滑太一致，被攻擊者的逐點統計量吸收」這個假設，被D02四個點正面推翻。

**修正後的解讀**：訓練log顯示所有五個設定的confuse loss項最終都收斂到幾乎同一個值（≈5.598，非常接近log(256)=5.545的均勻分佈下限），代表**不管有沒有正則化，產生器都學會了把單一軌跡的預測分佈推向差不多同樣程度的「均勻化」**——這解釋了為什麼五個設定的PI也全部緊貼在-0.0945到-0.0974這個窄範圍內，不管GE差多少（111.84到197.76）：PI量的是單軌跡層級的confusion程度（全部設定接近），GE量的是這個confusion在幾千條軌跡的log-likelihood累加後，是否還殘留可被攻擊者利用的、跟正確金鑰相關的系統性訊號（差異巨大）。**沒有正則化的版本，confusion本身沒有更強，但拿掉約束後產生器學到的擾動形狀，反而讓這個殘留訊號更容易被多軌跡累加找出來**——正則化在這裡看起來不是「阻力」，而是幫助訓練找到一個不會留下這種可累加殘留訊號的解。

這也把附錄A.3裡「擾動被攻擊者的逐點Standardizer/MinMax部分吸收」這個機制猜測，重新導向一個更根本的結構性解釋：這整個產生器家族（不論正則化強弱）都是**輸入軌跡的確定性函數，同一條軌跡每次都會得到同一個擾動**，不像高斯噪訊那樣對每條軌跡獨立注入隨機性——確定性擾動的某個成分本質上可能是「可被攻擊者的固定判別函數部分預測/抵銷」的，而獨立隨機噪訊天生做不到這件事。D01 vs 高斯基準線的效率差距，可能主要來自「確定性 vs 隨機性」這個結構性差異，而不是loss權重這個維度——這點loss權重掃描本身無法回答（五個設定都是確定性函數），需要另外設計實驗（例如在產生器輸出後疊加一層可控的隨機成分，或比較「同一模型多次inference是否給出相同擾動」跟「打亂/重跑seed是否改變攻擊者-可利用性」）才能驗證。

### A.6 D03：確定性 vs 隨機性假說，三點同樣被推翻

`src/generator/conv_perturber.py::AlwaysOnStochasticNoise`——跟Keras內建的`GaussianNoise`不同，這個層不管`training`旗標為何，每次呼叫都真的採樣新的雜訊並疊加到tanh-bounded delta上（再clip回[-1,1]），確保同一條軌跡在訓練或推論時每次呼叫都得到不同擾動，不是輸入的純函數。以D01（唯一勝過隨機基準的滿正則化配方）為基礎，疊加三個`noise_std`：

| 設定 | noise_std | PSR mean | GE@9000 | PI |
|---|---|---|---|---|
| **D01（基準，noise_std=0）** | 0 | **0.0249** | **111.84** | -0.0945 |
| D03_noise_low | 0.05 | 0.0341 | 119.54 | -0.0960 |
| D03_noise_mid | 0.15 | 0.0610 | 138.30 | -0.0995 |
| D03_noise_high | 0.30 | 0.1072 | 137.14 | -0.1034 |

**又是同一個模式**：noise_std越大，PSR成本越高，但GE也越差（0.05時119.54還壓在隨機基準127.5以下，0.15/0.30已經超過127.5、比什麼都不做還差）。**「確定性擾動缺乏隨機性」這個附錄A.5提出的假說，被這三點正面推翻**——加入獨立隨機性不但沒有幫助，反而讓效果單調變差，跟D02拿掉正則化的結果是同一種模式。

### A.7 兩個假說接連被推翻後的新假說：這可能是「對抗樣本」而非「資訊破壞」的問題

D02（四點）+ D03（三點），總共七個偏離D01設定的變體，**方向各不相同（放鬆正則化 vs 注入隨機性），但結果完全一致**：任何偏離D01那個精確、緊約束、平滑、確定性的解，PSR成本都上升，GE也都變差。這個一致性本身就是一個線索——問題可能不在「這個loss權重」或「有沒有隨機性」這類表層旋鈕，而在**這整個訓練目標的本質**。

`train_improved_defender.py`（及本專案原封不動移植的版本）的confuse loss，是標準的**對抗樣本（adversarial example）攻擊手法**：對一個凍結分類器，用梯度下降找出能讓輸出偏離正確標籤的最小擾動。這類方法的已知特性（ML安全文獻的常見觀察）是：梯度找到的擾動通常是**針對這一個特定模型的決策邊界的局部、脆弱捷徑**，而不是真的破壞輸入裡的底層資訊——換句話說，D01訓練出來的擾動可能只是學會了「怎麼讓E01這個特定攻擊者在單一次query上分類錯誤」，但底層的洩漏資訊（SNR意義上的訊號）可能大部分還在，只是被巧妙地移到了決策邊界的另一側。攻擊端的100-run log-likelihood累加評估法，正是設計來戳破這種「單次query被騙、但資訊還在、累加多次query後又能反推出來」的假象——這跟附錄A.3觀察到的「PI（單軌跡confusion程度）幾乎不變、GE（多軌跡累加後的可利用性）卻天差地遠」完全吻合。相對地，高斯噪訊完全不管任何模型的決策邊界，暴力地對每個時間點注入獨立雜訊，直接推高類內變異數、壓低SNR分母——這是在破壞底層資訊本身，不是在鑽某個模型的決策邊界漏洞，所以不管換哪個攻擊者、累加多少次query，效果都穩定。

**這個假說是可以直接驗證的，且工具已經現成**：攻擊端`src/metrics/leakage.py::snr`/`nicv`可以直接對「乾淨E集」vs「D01防禦後的E集」算逐點SNR，跟同PSR的高斯噪訊對SNR的影響做對照。如果D01的擾動幾乎不影響SNR峰值（訊號還在），而高斯噪訊在同PSR下明顯壓低SNR峰值，就直接證實這個假說；如果D01的擾動同樣有效壓低SNR，代表問題出在別的地方，需要重新想。

### A.8 SNR驗證：A.7的簡單版本被推翻，但揪出了更精確的真正機制——洩漏點覆蓋不完整

新增 `dlsca-defense-v2/scripts/03_check_snr_reduction.py`：不需要重新訓練，直接讀已有的`defended_traces.npy`（GAN跟高斯，同PSR成本），用攻擊端`src/metrics/leakage.py::snr`（已有、已測試過的純函式）比較「防禦前後」的逐點SNR下降幅度。

**第一次嘗試（ID標籤，即E01實際訓練的目標）：方法本身有問題，作廢**——乾淨E集在ID（未遮罩）標籤上的SNR峰值只有0.0335，本身就已經在雜訊層級（ASCAD遮罩機制設計上就是要讓一階洩漏在未遮罩標籤上趨近於0，這也是為什麼E01非得靠CNN學高階/多變量組合才能破解——見攻擊端`dlsca-attack-v2` CLAUDE.md附錄B.11對masked/unmasked標籤差異的討論）。高斯噪訊防禦後這個值甚至還「上升」了2.9%，這種雜訊層級的波動不具任何意義。**用univariate ID-SNR測A.7，metric本身選錯了，第一次結果不採信。**

**改用 ID_MASKED（mask_index=0，遮罩已知標籤，physically真實存在的洩漏點）重新測，在遮罩值本身的洩漏點（point 517）**：

```
乾淨E集 峰值SNR: 6.6787 @ point 517
GAN防禦後:        4.4535 @ point 514（-33.3%，PSR=0.0249）
高斯防禦後:       4.5303 @ point 517（-32.2%，PSR=0.0223，成本還比GAN低）
```

**GAN在這個點上壓低SNR的幅度，跟高斯噪訊幾乎一樣（甚至略多），成本還略高一點點——A.7原本「GAN沒有真的壓低SNR」這個簡單版本的假說，在這個洩漏點上被推翻。**

**但檢查第二個必要成分——遮罩值r_out本身的洩漏點（point 156，跟plaintext/key無關，純粹測r_out是否直接洩漏在某個時間點）——差異就出現了**：

```
乾淨E集 峰值SNR: 1.3316 @ point 156
GAN防禦後:        1.2751 @ point 182（-4.2%，峰值位置還偏移了，代表GAN根本沒有針對這個點）
高斯防禦後:       1.1790 @ point 156（-11.5%，峰值位置沒變，代表這裡被均勻地壓低了）
```

**高斯噪訊對這個點的壓制幅度是GAN的2.7倍**，且高斯的峰值位置完全沒有偏移（因為它是逐點獨立、無差別地對整條波形加噪，天生就會同時打到所有洩漏點），而GAN的峰值位置從156偏移到182，代表GAN的擾動根本沒有集中在這個真正的洩漏點上。

**修正後的機制解讀（比A.7原本的說法更精確、更有證據支撐）**：ASCAD這類遮罩防禦需要攻擊者同時掌握**兩個**東西才能破解——遮罩後的值（point 517）跟遮罩本身的值（point 156），E01光靠ID標籤訓練就能破解，代表它學到的是某種結合這兩個點的高階/多變量組合。**GAN訓練出來的擾動，把有限的擾動預算幾乎全部押在point 517上（在那裡表現跟高斯打平甚至略好），卻幾乎沒有動到point 156**——這很可能是因為訓練用的confuse loss只在乎「讓凍結攻擊者E01這一個特定模型在單次query上出錯」，梯度下降找到的捷徑剛好集中在對E01決策邊界最敏感的那個點，沒有動機去顧及point 156這個「攻擊者理論上也需要、但可能不是梯度最敏感方向」的第二洩漏點。高斯噪訊沒有這種「挑最敏感方向」的偏好，逐點獨立无差別加噪，天生會同時覆蓋兩個洩漏點——這正是D02、D03七次系統性嘗試（不管怎麼調loss權重或加隨機性，都沒能讓GAN打贏）背後真正的結構性原因：**問題不在「夠不夠隨機」或「夠不夠平滑」，而在「梯度下降找到的攻擊捷徑，天生傾向於只覆蓋部分洩漏點，而不是像高斯噪訊那樣無差別覆蓋全部洩漏點」**。

### A.9 D04設計：可微分的洩漏抑制loss（`src/metrics/leakage_loss.py`）

**設計目標**：把loss從「只讓E01單次query出錯」換成「同時壓低所有已知洩漏點的可分性」，直接對應A.8診斷出的機制。

**關鍵設計決策——為什麼用CPA式相關係數，不用真正的批次內class-conditional SNR**：`src/metrics/leakage.py::snr`需要對256個類別分別算類內變異數，batch_size=64的話大多數類別在一個batch裡只有0-1個樣本，統計量不穩定、不能用。改用**平方Pearson相關係數**（defended trace逐點跟label的Hamming weight代理之間）——這是CPA（Correlation Power Analysis）的標準做法，物理假設是「功耗與處理資料的Hamming weight近似線性相關」，這個假設讓CPA能用遠少於SNR/NICV所需的樣本數就估出穩定的相關係數，是SCA領域行之有年的權衡（線性模型假設換取小樣本可用性；如果真實洩漏不是線性於HW，這個loss項會跟CPA本身一樣有盲點——這點需要明確記錄）。

**Soft-max（logsumexp）而非硬max或mean**：對整條軌跡700個點的平方相關係數取硬max只會在當前最強的那一個點給梯度（容易「打地鼠」，這一步壓下去、換一個點冒出來，訓練不穩定）；取mean會被699個不相關的點稀釋掉真正重要的1-2個點的梯度訊號。Soft-max用temperature參數在兩者之間取得平衡，同時對「目前偏高的所有點」都給出有意義的梯度，且處處可微。

**新增檔案**：`src/metrics/leakage_loss.py`（`hamming_weight`/`pointwise_squared_correlation`/`soft_peak_leakage`）、`src/leakage_probe.py`（從metadata算出兩個洩漏頻道的label：`masked_value`＝Sbox[p⊕k]⊕mask，`mask_value`＝mask本身；`AES_SBOX`從攻擊端複製過來，不跨repo import，同`perturbation.py`的既有原則）。`src/train/adversarial.py::fit()`新增`leakage_labels`/`lambda_leakage`/`leakage_temperature`三個參數，**`lambda_leakage=0.0`（預設）完全不計算這個loss項，不是零權重加總**——確保D01/D02/D03既有config的行為逐位元不變。`configs/base.yaml`新增對應預設值（含`leakage_probe.enabled: false`開關）。6個新測試（`test_leakage_loss.py`）+ 1個新測試（`test_leakage_probe.py`）+ 2個`test_adversarial.py`新測試，`tests/`現在36個測試全過。

**設計過程中發現的一個重要理論限制（寫測試時才浮現，值得記錄）**：第一版驗證測試原本想確認「對一個有洩漏的欄位直接做梯度下降，能不能讓這個loss下降」，用一個**可訓練的仿射擾動**（純量縮放/平移，不依賴輸入）去優化，結果loss完全不動——诊断後發現這是**數學上必然的**：Pearson相關係數對「被觀察變數的任何非零仿射變換」不變（`corr(a·x+b, y) = sign(a)·corr(x,y)`，跟`a`、`b`的大小完全無關，只有正負號可能翻轉）。這代表：**純粹縮放或平移一個洩漏欄位，不管縮放係數多小，都不會讓這個loss下降**（只有在縮放係數趨近於0、也就是幾乎把整個欄位歸零時，才會因為分母的極小值保護項而讓相關係數退化到0）。测试改成讓可訓練變數控制**注入的獨立雜訊量**（不是縮放/平移，是真正加入額外變異）後，loss如預期正確下降——**這證實了這個loss真正獎勵的是「注入獨立雜訊」這類手法，如果只是仿射變換洩漏欄位是無效的**。這對D04的產生器設計有直接意義：D01的產生器目前是純確定性函數（`noise_std=0`），如果它學到的解只是對洩漏欄位做簡單縮放/平移，這個新loss不會有效果；CNN要嘛得學會利用其他時間點的資訊建構出一個「相對於label是隨機的」訊號（架構上可行，因為Conv1D對整條700點軌跡都有感受野），要嘛就需要搭配D03已經做好的`AlwaysOnStochasticNoise`（`generator.noise_std>0`）提供真正的隨機性來源——這是第一次D04實驗設計時特意先只測純確定性版本（`configs/exp/D04_snr_aware.yaml`保持`noise_std=0`）的原因：先確認CNN的跨欄位資訊利用能力夠不夠，不夠的話再加隨機性，兩個變數不要一次疊加，才知道效果歸功於誰。

**已完成端到端smoke test**（小規模：300條軌跡、2 epoch，對真實E01攻擊者）：確認`leakage_probe.enabled=true`能正確從metadata算出兩個洩漏頻道的label，訓練迴圈正確把`leak_masked_value`/`leak_mask_value`兩項算進total loss並記錄進`train_history.csv`，管線接線正確。**尚未跑正式的15000條軌跡/40epoch GPU實驗**——`configs/exp/D04_snr_aware.yaml`已經寫好（`lambda_leakage=0.05`，其餘沿用D01），下一步是實際跑一次並用攻擊端Stage B介面做100-run正式評估。

### A.10 下一步（留待後續決定）

- **正式跑D04**：`configs/exp/D04_snr_aware.yaml`（GPU，15000條/40epoch），完成後用攻擊端Stage B介面正式評估、跟D01（GE@9000=111.84）及同PSR高斯基準線比較。
- 如果D04的GE明顯優於D01但還是打不贏高斯基準線，下一步是重新跑一次`--override generator.noise_std=0.05`（或類似）疊加D03已驗證過的隨機性層，驗證上一段提到的「CNN跨欄位能力不夠、需要真隨機性搭配」這個備案假設。
- 如果D04本身就找到一個能打平/贏過高斯基準線的PSR點，才進入D05（原D04，三方比較圖）。

### A.11 TVLA檢查——方向.md的預測完全命中，「爆表」不是比喻

根據 `B11209017/方向.md` claim①的建議，新增 `scripts/04_check_tvla.py`：對兩個已知洩漏頻道（`masked_value`、`mask_value`，跟SNR診斷附錄A.8同一組）逐bit做Welch t-test（`dlsca-attack-v2/src/metrics/leakage.py::t_test`，跟SNR診斷同一種cross-repo引用方式），掃過8個bit取最壞（|t|最大）的那個，對照TVLA慣例門檻`|t|>=4.5`。

**ASCAD的Attack_traces沒有重複相同明文的「fixed」採集組**，所以做不了教科書等級的fixed-vs-random明文TVLA；改用SCA領域常見的替代做法——「specific」TVLA：用已知中間值的某個bit把軌跡切成兩組再做t檢定，等於單bit版的DPA/leakage detection，跟SNR/NICV同樣是model-agnostic（不涉及任何訓練模型）。

**結果**（D01 vs 高斯σ=0.25，PSR≈0.023-0.025同一組防禦後波形）：

```
channel: masked_value（point 517附近）
  clean（正對照組）    worst |t| = 71.73   -> LEAKAGE DETECTED
  GAN-defended         worst |t| = 67.61   -> LEAKAGE DETECTED
  Gaussian-defended    worst |t| = 68.80   -> LEAKAGE DETECTED

channel: mask_value（point 149附近）
  clean（正對照組）    worst |t| = 52.82   -> LEAKAGE DETECTED
  GAN-defended         worst |t| = 52.98   -> LEAKAGE DETECTED（比clean還高！）
  Gaussian-defended    worst |t| = 50.70   -> LEAKAGE DETECTED
```

**跟方向.md的預測幾乎逐字命中**：「這個實驗很可能給你們負面結果...洩漏根本沒減少，只是位置被搬走了...TVLA大概率仍然爆表」——確實爆表，而且不是差一點點：門檻是4.5，兩個防禦、兩個頻道量出來的|t|全部落在50-72之間，是門檻的11到16倍，GAN跟Gaussian從clean的71.73/52.82只降到67-69/51-53，降幅在5-10%之間，遠遠不足以讓|t|掉到門檻以下。**mask_value頻道上GAN甚至比clean的|t|還高**，等於這個bit的可分性完全沒有被觸碰。

**這個結果比SNR診斷（附錄A.8）更嚴厲，也補上了一塊SNR沒講清楚的東西**：SNR診斷看的是「峰值降了多少百分比」（GAN在masked_value降了33%，看起來像是有實質壓制），但TVLA用的是「絕對可分性夠不夠低到偵測不到」這個security-evaluation的標準二元判準——即使SNR峰值降了三成，剩下的七成在TVLA的尺度下仍然是壓倒性的洩漏，跟「有沒有真的防住」完全是兩回事。**這正是方向.md claim①一開始就講的不對稱：安全性主張只能被證偽，做SNR/PI這類連續指標容易讓人誤以為「有改善=有防護」，TVLA用的是攻防評估慣用的二元門檻，比較不會製造這種錯覺。**

**這對整個防禦敘事的意義**：不只是「GAN輸給高斯」，是「在我們目前測過的PSR成本範圍內，GAN跟高斯這兩種手法都沒有真正達到TVLA意義下的安全——它們都只是讓某一個特定攻擊模型的single-query分類變困難，底層的model-agnostic洩漏幾乎原封不動」。這比D01-D04單純比較「誰的GE比較低」更根本，也更貼近方向.md推薦的Route A論述核心（「失效是資訊理論的，不是工程的」）——現在有TVLA這個獨立、model-agnostic的證據支撐這句話，不只是靠SNR的間接推論。

**尚未做、留待後續**：目前只測了一個PSR點（高斯σ=0.25），還沒掃過附錄C.3整條高斯曲線去看TVLA會不會在更高成本（例如σ=1.5，那個讓GE完全不收斂的門檻點）真的通過門檻——如果連σ=1.5都還是爆表，代表「用等功率噪訊/擾動達到TVLA意義下的安全」在這個資料集上的成本可能高到不切實際，這本身會是一個值得寫進報告的量化結論。
