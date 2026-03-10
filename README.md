# Dream to Control — World Model Comparison

基於 [DreamerV1](https://github.com/danijar/dreamer) 的擴展實驗，比較三種世界模型在低維狀態環境（Pendulum）的表現。

> **Note:** 查看 [DreamerV2](https://github.com/danijar/dreamerv2) 以獲得支援 Atari 和 DMControl 的更新版本。

<img width="100%" src="https://imgur.com/x4NUHXl.gif">

## 比較的世界模型

| 模型 | 說明 | 輸入 |
|------|------|------|
| **RSSM** | 原版 DreamerV1，CNN 編碼圖像 + 遞迴隨機狀態模型 | 圖像（64×64） |
| **LinearSSM** | 線性動態模型 `next_obs = A @ obs + B @ action`，最小二乘法擬合 | 狀態向量 |
| **MLPDynamics** | 兩層 MLP 預測下一狀態，MSE loss + SGD 訓練 | 狀態向量 |

**核心問題：** 對於低維狀態的簡單環境，RSSM 的複雜度是否有必要？

## 新增檔案

- `alt_models.py` — LinearSSM 與 MLPDynamics 實作，與 RSSM 相同介面
- `evaluate.py` — 評估腳本，產生 reward curve 與 prediction MSE 圖表
- `run_comparison.sh` — 一鍵跑完三個模型並產生比較圖

## 環境安裝（Apple Silicon / macOS）

```bash
conda create -n dreamer python=3.9 -y
conda activate dreamer
pip install tensorflow-macos==2.8.0 tensorflow-metal==0.4.0
pip install tensorflow-probability==0.16.0
pip install "gym==0.25.2" Pillow matplotlib "numpy<2"
```

## 執行方式

**跑全部三個模型並比較：**
```bash
bash run_comparison.sh
```

**只跑單一模型：**
```bash
bash run_comparison.sh rssm    # 原版 Dreamer RSSM
bash run_comparison.sh linear  # LinearSSM
bash run_comparison.sh mlp     # MLPDynamics
```

**或直接用 Python：**
```bash
python dreamer.py \
    --task gym_Pendulum-v1 \
    --world_model linear \   # rssm | linear | mlp
    --logdir logs/linear \
    --steps 50000 \
    --prefill 1000 \
    --precision 32 \
    --log_images False
```

**評估並產生圖表：**
```bash
python evaluate.py \
    --logdirs logs/rssm logs/linear logs/mlp \
    --labels RSSM LinearSSM MLP \
    --outdir results
```

結果圖表存於 `results/`：
- `reward_curves.png` — 各模型訓練過程的 episode return
- `prediction_mse.png` — 5-step 預測 MSE

## 原版 Dreamer 使用方式

訓練（DMControl）：
```bash
python dreamer.py --logdir ./logdir/dmc_walker_walk/dreamer/1 --task dmc_walker_walk
```

TensorBoard：
```bash
tensorboard --logdir ./logdir
```

## 引用

```bibtex
@article{hafner2019dreamer,
  title={Dream to Control: Learning Behaviors by Latent Imagination},
  author={Hafner, Danijar and Lillicrap, Timothy and Ba, Jimmy and Norouzi, Mohammad},
  journal={arXiv preprint arXiv:1912.01603},
  year={2019}
}
```

---

原始碼來自 [danijar/dreamer](https://github.com/danijar/dreamer)，本 repo 在其基礎上新增世界模型比較實驗。
