# README 成果動畫（Manim Community）

`../results.gif` 由 `results_scene.py` 產生；所有數字經 `results_data.py` 從
`reports/phase4/full/public/phase4-results.json` 與
`reports/tmmlu-v1.1/20260913T170025Z/analysis.json` 讀取，不手填。
`tests/test_media_results_data.py` 會檢查數字與證據一致，且腳本內沒有手打的結果數字。

Manim 與 ffmpeg 不屬於專案相依，不進 `pyproject.toml`、lock 與 CI；用 `uvx` 以獨立工具環境執行。

## 重新產生

需求：`uv`、PATH 上有 `ffmpeg`、系統字型「Microsoft JhengHei」（Windows 內建）。不需要 LaTeX。
Manim 以 Python 3.12 執行（3.13 以上部分二進位套件尚無 wheel）。

```powershell
uvx --python 3.12 manim -r 960,540 --fps 15 --media_dir media/manim/.media media/manim/results_scene.py Results
ffmpeg -y -i media/manim/.media/videos/results_scene/540p15/Results.mp4 -vf "fps=12,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" -loop 0 media/results.gif
```

第一步輸出 MP4；第二步用 ffmpeg 產生調色盤最佳化的 GIF（Manim 直接輸出的 GIF 約 27 MB，
不適合放 README）。`media/manim/.media/` 為渲染暫存，已列入 `.gitignore`。
