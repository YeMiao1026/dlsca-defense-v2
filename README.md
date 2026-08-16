# dlsca-defense-v2

用 GAN／對抗訓練式擾動產生器訓練旁通道防禦，並用**成本匹配**的方式證明它比單純加高斯噪訊或時域干擾更有效率。是「基於生成對抗網路之主動式對抗旁通道防禦機制」專題的防禦端子系統，攻擊端另案 [`dlsca-attack-v2`](https://github.com/YeMiao1026/dlsca-attack-v2)。

完整計畫、威脅模型、跟攻擊端的介面定義見 [`CLAUDE.md`](./CLAUDE.md)。

**目前狀態：D01＋D02＋D03已跑完，兩個假說接連被推翻**。D01的誠實結果是負面的：跟同PSR成本的高斯噪訊基準線比，移植的GAN產生器效率明顯更差（`GE@9000=111.84` vs 高斯噪訊在同PSR下 `GE@9000=0.00`）。D02猜測是loss的smooth/l2懲罰把擾動壓得太平滑太一致——四點掃描（拿掉/降低這兩項懲罰）結果相反：正則化越弱，成本越高、GE也越差。D03接著測「確定性擾動缺乏隨機性」這個假說（疊加每次inference獨立採樣的雜訊）——三點同樣是成本越高、GE越差，假說再度被推翻。七次系統性嘗試（D02四點+D03三點）方向各異，結果卻高度一致，新的懷疑方向是：D01訓練的可能是傳統意義上的「對抗樣本」（讓攻擊者單次query出錯的局部捷徑），而不是真正壓低SNR的「資訊破壞」，這跟高斯噪訊的作用機制本質不同——下一步是直接用攻擊端的SNR工具驗證這個假說。詳見 `CLAUDE.md` 附錄A（文件末尾）跟 [`docs/runs.md`](./docs/runs.md)。

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
