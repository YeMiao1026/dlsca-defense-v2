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

## TVLA全高斯曲線掃描

同一套TVLA對附錄C.3全部8個高斯sigma_ratio點各跑一次：

| sigma_ratio | PSR | masked_value \|t\| | mask_value \|t\| |
|---|---|---|---|
| 0.1 | 0.0089 | 71.27 | 52.32 |
| 0.25 | 0.0223 | 68.80 | 50.70 |
| 0.5 | 0.0446 | 61.66 | 46.35 |
| 0.75 | 0.0668 | 53.58 | 40.96 |
| 1.0 | 0.0891 | 46.33 | 35.85 |
| 1.5 | 0.1337 | 35.51 | 27.85 |
| 2.0 | 0.1783 | 28.45 | 22.46 |
| 3.0（最高成本點） | 0.2674 | 20.28 | 16.11 |

連PSR最高的點都還差門檻3.5-4.5倍。log-log冪律外插估計要 sigma_ratio≈294-396 才會壓到門檻以下——這個外插範圍遠超實測（0.1-3.0），精確度不能太當真，但量級本身（要淹沒訊號到近300倍原始振幅）已經足以說明：單純加噪訊要達到TVLA級安全，成本高到不切實際。詳見 `CLAUDE.md` 附錄A.12。

## D03注入熵直接測量（`scripts/05_measure_injected_entropy.py`）

不重跑訓練，對同一批固定的乾淨軌跡重複呼叫已訓練好的D03 `generator.keras` 30次，測量兩個已知洩漏點（`masked_value` point 517、`mask_value` point 156）上的重複間變異數，換算成bits：

| run | noise_std | masked_value bits | mask_value bits | GE@9000（已有數據） |
|---|---|---|---|---|
| D01 | 0 | -inf（精確0，正對照組） | -inf | 111.84 |
| D03_noise_low | 0.05 | 0.31 | 0.22 | 119.54 |
| D03_noise_mid | 0.15 | 1.87 | 1.90 | 138.30 |
| D03_noise_high | 0.30 | 2.84 | 2.86 | 137.14 |

理論值（noise_std×epsilon）幾乎完全對上實測std，確認熵確實有被注入且打在對的點上。**但熵單調上升、GE卻單調變差**——不是「隨機性沒打對地方」，是打對了地方也沒用，原因尚未解開，誠實記錄為開放問題。詳見 `CLAUDE.md` 附錄A.13。

## D05（設計完成，尚未正式跑）

`src/generator/universal_perturber.py::UniversalTemplate` 新增通用擾動模板架構（單一可訓練向量、輸出跟輸入軌跡內容無關，對每一條軌跡廣播同一個學出來的波形），跟`conv_perturber.py`的CNN版共用完全相同介面，靠`generator.architecture: cnn|universal`設定分派。`configs/exp/D05_universal.yaml`已寫好（沿用D01其餘設定）。6個新測試，`tests/`現在42個測試全過；小規模smoke test（300條/3epoch，真實E01攻擊者）確認管線接線正確。**尚未跑正式15000條/40epoch GPU實驗**，詳見 `CLAUDE.md` 附錄A.14。
