# dlsca-defense-v2

用**成本匹配**的方式系統性、誠實地再評估對抗式擾動類SCA防禦（learned generator／universal perturbation／高斯噪訊）——不是預設GAN會贏，是把各種手法放在同一套嚴謹評估（100-run GE、TVLA、SNR）下公平比較，包括誠實報告負面結果。是「基於生成對抗網路之主動式對抗旁通道防禦機制」專題的防禦端子系統，攻擊端另案 [`dlsca-attack-v2`](https://github.com/YeMiao1026/dlsca-attack-v2)。方向依據 B11209017/方向.md 的Route A建議修正，見 `CLAUDE.md` §1/§8。

完整計畫、威脅模型、跟攻擊端的介面定義見 [`CLAUDE.md`](./CLAUDE.md)。

**目前狀態：D01-D03＋SNR驗證找到真正機制，D04（SNR-aware loss）設計完成、smoke test通過，尚未正式跑**。D01的誠實結果是負面的：跟同PSR成本的高斯噪訊基準線比，移植的GAN產生器效率明顯更差（`GE@9000=111.84` vs 高斯噪訊在同PSR下 `GE@9000=0.00`）。D02（loss權重掃描）跟D03（隨機性注入）兩個假說都被系統性推翻。用`scripts/03_check_snr_reduction.py`比較SNR壓制幅度後找到具體機制：ASCAD遮罩機制需要攻擊者同時掌握兩個洩漏點（遮罩後的值＋遮罩本身），**GAN在其中一個點壓得跟高斯噪訊一樣好甚至更好，但在另一個點幾乎沒動；高斯噪訊逐點無差別加噪，天生同時覆蓋兩者**。D04因此在loss裡新增一個可微分的洩漏抑制項（`src/metrics/leakage_loss.py`，CPA式平方相關係數＋soft-max，直接對兩個已知洩漏點取代單純「讓攻擊者單次query出錯」的目標），已完成設計、實作、36個測試、對真實E01攻擊者的小規模smoke test，`configs/exp/D04_snr_aware.yaml`已就緒，**下一步是正式跑15000條/40epoch的GPU實驗**。

**另外用 `scripts/04_check_tvla.py`（根據B11209017/方向.md claim①）補上了bit-split TVLA檢查**：兩個防禦（D01、高斯σ=0.25，同PSR成本）在兩個已知洩漏頻道上全部爆表（|t|=50-72，門檻4.5的11-16倍，比clean只降5-10%）——不只是GAN輸給高斯，是兩種手法在這個成本範圍內都沒有達到TVLA意義下的安全，只讓特定攻擊模型的單次分類變困難，底層model-agnostic洩漏幾乎沒動。這比GE比較更根本，也是Route A論述（「失效是資訊理論的，不是工程的」）的直接證據。掃過附錄C.3整條高斯PSR曲線後，連成本最高的點（sigma_ratio=3.0，噪訊振幅是軌跡本身振幅27%）都還差門檻3.5-4.5倍，冪律外插估計要sigma_ratio≈300-400（外插範圍遠超實測，精確度有限，但量級已經說明問題）——單純加噪訊要達到TVLA級安全，成本高到不切實際，這是支撐「需要真正masking而非hiding-only防禦」的量化證據。

**另外用 `scripts/05_measure_injected_entropy.py` 直接測量D03三個配方實際注入的熵**（不重跑訓練，對同一批固定軌跡重複呼叫已訓練好的generator 30次，量測重複間變異數）：理論值（noise_std×epsilon）幾乎完全對上實測，確認熵確實有被注入、而且確實打在兩個已知洩漏點上。但熵從0.3bits升到2.85bits，GE卻從119.54單調變差到137.14——不是「隨機性沒打對地方」，是打對了地方也沒讓攻擊變難，原因尚未解開，誠實記錄為開放問題而非硬套解釋。

**D05（通用擾動模板，Gu et al. 2020風格）設計完成、smoke test通過，尚未正式跑**：`src/generator/universal_perturber.py` 新增單一可訓練向量的架構（輸出跟輸入軌跡內容無關，對每條軌跡廣播同一個學出來的波形），跟D01-D04的CNN版共用完全相同介面，靠`generator.architecture: cnn|universal`分派——同時是Route A需要比較的第三種防禦類別，也是方向.md claim②「工程成本比masking低」唯一站得住腳的架構（部署時只需疊加固定波形，不需要每次都跑CNN推論）。`configs/exp/D05_universal.yaml`已就緒，**下一步是正式跑15000條/40epoch的GPU實驗**。

詳見 `CLAUDE.md` 附錄A（文件末尾）跟 [`docs/runs.md`](./docs/runs.md)。

## 跟攻擊端的關係

不共用程式碼，只共用資料格式：防禦端輸出一個 `defended_traces.npy`（跟攻擊端 `attack_traces[split.e]` 同shape），呼叫攻擊端既有的 Stage B 評估介面（`02_run_attack.py --traces/--out` + `03_evaluate.py --probs/--out`）拿到嚴謹的 100-run GE/N_TGE 結果，不自己另外寫評估邏輯。細節見 CLAUDE.md §2。

## 環境需求

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

需要能存取一份已經訓練好的 `dlsca-attack-v2` 攻擊者（`runs/{exp}/model.keras` + `config_snapshot.yaml` + `split_indices.npz`）跟同一份 ASCAD 資料。

## 測試

```bash
python3 -m pytest tests/ -q
```
