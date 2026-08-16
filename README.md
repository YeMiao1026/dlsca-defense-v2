# dlsca-defense-v2

用 GAN／對抗訓練式擾動產生器訓練旁通道防禦，並用**成本匹配**的方式證明它比單純加高斯噪訊或時域干擾更有效率。是「基於生成對抗網路之主動式對抗旁通道防禦機制」專題的防禦端子系統，攻擊端另案 [`dlsca-attack-v2`](https://github.com/YeMiao1026/dlsca-attack-v2)。

完整計畫、威脅模型、跟攻擊端的介面定義見 [`CLAUDE.md`](./CLAUDE.md)。

**目前狀態：D01-D03＋SNR驗證已跑完，找到真正機制**。D01的誠實結果是負面的：跟同PSR成本的高斯噪訊基準線比，移植的GAN產生器效率明顯更差（`GE@9000=111.84` vs 高斯噪訊在同PSR下 `GE@9000=0.00`）。D02（loss權重掃描）跟D03（隨機性注入）兩個假說都被系統性推翻——不管往哪個方向調，結果都一致變差。最後用`scripts/03_check_snr_reduction.py`直接比較SNR壓制幅度，找到具體機制：ASCAD遮罩機制需要攻擊者同時掌握兩個洩漏點（遮罩後的值＋遮罩本身），**GAN在其中一個點壓制得跟高斯噪訊一樣好、甚至更好，但在另一個點幾乎沒動；高斯噪訊逐點無差別加噪，天生同時覆蓋兩者**——這才是七次系統性掃描都打不贏高斯基準線的真正原因，不是超參數沒調對，是loss目標本身只顧著騙過凍結攻擊者的單次輸出，沒有要求同時破壞所有已知洩漏點。詳見 `CLAUDE.md` 附錄A（文件末尾）跟 [`docs/runs.md`](./docs/runs.md)。

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
