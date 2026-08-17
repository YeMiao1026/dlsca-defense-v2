# dlsca-defense-v2

用**成本匹配**的方式系統性、誠實地再評估對抗式擾動類SCA防禦（learned generator／universal perturbation／高斯噪訊）——不是預設GAN會贏，是把各種手法放在同一套嚴謹評估（100-run GE、TVLA、SNR）下公平比較，包括誠實報告負面結果。是「基於生成對抗網路之主動式對抗旁通道防禦機制」專題的防禦端子系統，攻擊端另案 [`dlsca-attack-v2`](https://github.com/YeMiao1026/dlsca-attack-v2)。方向依據 B11209017/方向.md 的Route A建議修正，見 `CLAUDE.md` §1/§8。

完整計畫、威脅模型、跟攻擊端的介面定義見 [`CLAUDE.md`](./CLAUDE.md)。

**目前狀態：21天路線圖（TVLA、D03熵重分析、D05通用擾動模板、D06自適應攻擊者）四個核心項目全部完成，報告草稿已整理完畢。** 核心結論：D01（移植的GAN產生器）效率明顯不如同PSR成本的高斯噪訊基準線；D02/D03共七次系統性loss權重/隨機性掃描全數推翻兩個直覺假說；SNR/TVLA診斷找出真正機制（洩漏點覆蓋不完整），且顯示兩種防禦在測過的成本範圍內都沒有真正達到TVLA意義下的安全；D05證明「GE看起來贏」可以跟真正的資訊破壞可證明地脫鉤；D06證明凍結攻擊者這個威脅模型假設過於樂觀——自適應重訓練幾乎完全瓦解D01的防禦（N_TGE只比完全無防禦多25%）。

完整敘事、圖表、方法論貢獻整理見報告草稿；逐項實驗的詳細記錄見 `CLAUDE.md` 附錄A–E（文件末尾）跟 [`docs/runs.md`](./docs/runs.md)。

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
