# 執行紀錄

索引 `runs/`（git-ignored）底下的實際執行結果。詳細分析見 `CLAUDE.md` 附錄 A。

| run_dir | 攻擊者 | n_train | epochs（實際/上限） | lambda_l2 | lambda_smooth | PSR mean | GE@9000 | N_TGE | PI |
|---|---|---|---|---|---|---|---|---|---|
| `D01_replicate_baseline_20260816_232552_948723` | E01（GPU server） | 15000 | 34/40（early stop） | 0.005 | 0.002 | 0.0249 | 111.84 | None | -0.0945 |
| `D02_nosmooth_20260816_235627_964618` | E01（GPU server） | 15000 | 32/40（early stop） | 0.005 | 0.0 | 0.0393 | 137.32 | None | -0.0959 |
| `D02_lowreg_20260816_234849_958486` | E01（GPU server） | 15000 | 24/40（early stop） | 0.0005 | 0.0002 | 0.0636 | 194.83 | None | -0.0974 |
| `D02_nol2_20260816_235912_966985` | E01（GPU server） | 15000 | 40/40（未觸發早停） | 0.0 | 0.002 | 0.0974 | 187.70 | None | -0.0969 |
| `D02_noreg_20260816_234954_960370` | E01（GPU server） | 15000 | 17/40（early stop） | 0.0 | 0.0 | 0.1493 | 197.76 | None | -0.0974 |

D02 四點結論：拿掉/降低 l2、smooth 懲罰後，PSR成本全部上升（1.6x-6x），但 GE 也全部上升（比D01更差、多數比隨機基準127.5還差）——結論跟附錄A原本的假設相反，正則化不是阻力、反而是必要條件，見 `CLAUDE.md` 附錄A.5。

| run_dir | noise_std | PSR mean | GE@9000 | N_TGE | PI |
|---|---|---|---|---|---|
| `D03_noise_low_20260817_001038_981150` | 0.05 | 0.0341 | 119.54 | None | -0.0960 |
| `D03_noise_mid_20260817_001733_986381` | 0.15 | 0.0610 | 138.30 | None | -0.0995 |
| `D03_noise_high_20260817_001746_982744` | 0.30 | 0.1072 | 137.14 | None | -0.1034 |

D03（在D01基礎上疊加AlwaysOnStochasticNoise）三點結論：同D02一樣的模式——noise_std越大PSR成本越高，但GE也越差（0.05時119.54還在隨機基準以下，0.15/0.30已經超過127.5），「確定性擾動缺乏隨機性」這個假設也被推翻，見 `CLAUDE.md` 附錄A.6/A.7。

## SNR驗證（`scripts/03_check_snr_reduction.py`，不需重訓練）

比較 D01（`defended_traces.npy`，PSR=0.0249）vs 高斯噪訊 sigma_ratio=0.25（PSR=0.0223，來自攻擊端 `defenses/gaussian_sigma0.25_20260816_225339_137443/`）在兩個真實洩漏點上的SNR壓制幅度：

| 洩漏點 | 乾淨SNR峰值 | GAN防禦後 | 高斯防禦後 |
|---|---|---|---|
| 遮罩值 Z'=Sbox[p⊕k]⊕mask（point 517） | 6.68 | 4.45（-33.3%） | 4.53（-32.2%） |
| 遮罩本身 r_out（point 156，跟plaintext/key無關） | 1.33 | 1.28（-4.2%，峰值還偏移到182） | 1.18（-11.5%，峰值沒動） |

**GAN在第一個洩漏點壓得比高斯還兇，但在第二個洩漏點幾乎沒動**——這才是D01效率不如高斯噪訊的真正機制：梯度下降找到的擾動集中火力在對凍結攻擊者最敏感的那一個點，沒有覆蓋到攻擊者實際需要的另一個洩漏點；高斯噪訊逐點獨立無差別加噪，天生同時覆蓋兩者。詳見 `CLAUDE.md` 附錄A.8/A.9。

## D04（設計完成，尚未正式跑）

`src/metrics/leakage_loss.py` 新增可微分洩漏抑制loss（CPA式平方相關係數 + soft-max），`configs/exp/D04_snr_aware.yaml` 已寫好（`lambda_leakage=0.05`，其餘沿用D01，`noise_std=0`維持確定性）。小規模smoke test（300條/2epoch，真實E01攻擊者）確認管線接線正確，`tests/`36個測試全過。**尚未跑正式15000條/40epoch GPU實驗**，詳見 `CLAUDE.md` 附錄A.9/A.10。

## TVLA檢查（`scripts/04_check_tvla.py`，根據方向.md claim①）

Bit-split TVLA（8個bit逐bit Welch t-test取最壞值，門檻`|t|>=4.5`），對D01與高斯σ=0.25（PSR≈0.023-0.025）在兩個已知洩漏頻道上檢查：

| 頻道 | clean（正對照組） | GAN-defended | Gaussian-defended |
|---|---|---|---|
| masked_value | 71.73 → 洩漏 | 67.61 → 洩漏 | 68.80 → 洩漏 |
| mask_value | 52.82 → 洩漏 | 52.98 → 洩漏（比clean還高） | 50.70 → 洩漏 |

**全部爆表**，門檻4.5的11-16倍，降幅只有5-10%。跟方向.md的預測幾乎逐字命中——兩種防禦在這個PSR成本範圍內都沒有達到TVLA意義下的安全，只是讓特定攻擊模型的單次分類變困難，底層model-agnostic洩漏幾乎原封不動。詳見 `CLAUDE.md` 附錄A.11。
