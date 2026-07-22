# tw-med-llm-qlora 專案執行簿

最後更新：2026-07-22

## 研究規格

比較同尺寸原廠 instruct、台灣在地化 base、台灣 base + 醫療 QLoRA adapter，在 MedQA 台灣 test 測量專業能力增益，並以 TMMLU+ 醫學科與非醫學控制科檢查 catastrophic forgetting。訓練只在 Colab 執行；可重現程式與離線 handoff 已移轉至 Windows RTX 4090，並完成 base + adapter 推論驗收。

固定原則：

- 嚴格遵守 `AGENTS.md`。
- test 不得進入訓練、超參數選擇或 prompt 調整。
- 每個 Phase 展示證據並取得確認後才能前進。
- 任何付費批次工作先依 smoke/calibration 實測估價。
- 權重、原始題庫、完整私有輸出與 token 不進 Git。
- README 只填實測結果，不填預估成績。

## Phase 狀態與退出條件

| Phase | 狀態 | 退出條件 |
| --- | --- | --- |
| 0 骨架與工作流 | 完成 | uv lock、pytest、ruff、skill validation 通過；敏感檔未追蹤；使用者確認 |
| 1 資料驗證 | 完成 | 5 筆 gate、全量 audit、test 隔離、可重現 hash、pytest 通過；使用者確認 |
| 2 Smoke test | 完成 | 100 筆／10 steps 無 OOM/NaN，adapter 可重載，完成成本估算；使用者確認 |
| 3 完整訓練 | 完成 | checkpoint 可恢復、adapter/曲線/manifest 完整、validation 指標完成；使用者已確認跨 Phase |
| 4 雙軌評估 | 完成；28,758 requests、公開／私人封存與本機證據驗證均通過 | MedQA 三模型表、TMMLU+ 分科／穩定度／遺忘判定完成 |
| 5 本機與發布 | 進行中；step-700 adapter 與 RTX 4090 acceptance 已驗證，GitHub 公開目標已建立，等待 hosted CI、HF 私人目標與 receipt | 4090 推論通過、模型卡完成、使用者核准目標與可見性、GitHub hosted CI 通過、HF publication receipt 驗證通過 |

## 鎖定決策

| 日期 | 決策 | 理由 |
| --- | --- | --- |
| 2026-07-21 | README 技術章節保留必要模型與工具名稱 | 維持可重現性；動機不作商業敘事 |
| 2026-07-21 | MedQA 只報總體，分科交由 TMMLU+ | MedQA 台灣 split 沒有原生分科標籤 |
| 2026-07-21 | TMMLU+ 使用 8 醫學科 + 5 控制科 | 才能實際檢查通用能力退化 |
| 2026-07-21 | 三模型同場評估，只微調台灣模型 | 分離在地化效果與領域微調效果 |
| 2026-07-21 | 全量一個 seed，加每科至多 100 題三 seed | 平衡證據品質與 Colab 時間 |
| 2026-07-21 | 使用單一 repo-local workflow skill | 讓 Phase 規範隨 repo 版本化且避免 skill 過度拆分 |
| 2026-07-21 | MedQA 重複鍵採收斂空白後 casefold，不做 NFKC | NFKC 會折疊醫學符號並過度合併；保守規則可精確重現固定筆數 |
| 2026-07-21 | train/validation 排除缺值與選項文字重複題，test 不過濾 | 避免學入歧義目標，同時維持 test 1,413 筆原始順序與內容 |
| 2026-07-21 | LangChain 限定用於 Agent/RAG/應用層編排 | QLoRA 訓練與評估應直接使用 Unsloth、Transformers、PEFT、TRL 等官方原生 API |
| 2026-07-21 | Colab Pro 可優先使用 premium GPU | 使用者明確授權以較高 GPU 吞吐加速；保持模型、資料、LR、max sequence 與 effective batch 16 不變 |
| 2026-07-22 | Phase 3 優先使用 A100，先校準再解鎖長跑 | 使用者明確核准進入 Phase 3；A100 吞吐與 CU 不能沿用 L4 投影 |
| 2026-07-22 | 核准 A100 40GB、1 epoch 完整訓練 | 使用者確認隔日執行，且 CU 以完成訓練為優先；採已驗證的 5.00 小時／26.49 CU 投影與 31.78 CU 緩衝 |
| 2026-07-22 | 正式 adapter 選用 step 700 | 完整訓練到 step 703，但額外 step 703 validation 為 NaN；step 700 已通過全 1,409 筆 validation，且 recovery 的 checkpoint SHA-256、PEFT state 與重載皆通過 |
| 2026-07-22 | Phase 4 先以 20 題 TMMLU+ validation 校準三模型 | 先驗證 vLLM、adapter serving、Twinkle parser、吞吐與成本；不讓 test 參與 prompt 或執行參數調整 |
| 2026-07-22 | Twinkle Eval 由專案預先固定選項排列 | v2.8.0 內建 shuffle 未提供可設定 seed；先以 stable ID + seed 重排，再設 runner `shuffle_options=false` 才能保證三模型提示一致 |
| 2026-07-22 | Phase 4 完整工作量固定 28,758 次生成 | MedQA 三模型 4,239、TMMLU+ 全量三模型 16,719、穩定度 base/adapter 7,800；避免成本估算漏算 MedQA |
| 2026-07-22 | Colab 評估固定使用 vLLM 0.25.1 官方 `cu129` wheel | PyPI 的 plain wheel 在本次 Colab runtime 尋找 `libcudart.so.13` 而無法啟動；官方 release 提供可重現的 CUDA 12.9 wheel 與 SHA-256，並由 uv `--torch-backend=cu129` 配對 PyTorch |
| 2026-07-22 | 評估 parser 接受單一 A–D 或唯一簡單 boxed A–D | adapter 的訓練 target 是單一字母，而原廠模型常回 `\boxed{A}`；兩種皆是唯一且可嚴格判定的答案格式，缺答、多 box、無效或巢狀 box 仍拒絕 |
| 2026-07-22 | Phase 4 校準生成上限改為 256 tokens | 32-token 校準使台灣 base 的 20 筆輸出全在答案前截斷；修正版另要求各模型 parse rate ≥80% 且零 token-limit hit，未通過就不解鎖 test |
| 2026-07-22 | 256-token hit 計為答錯並獨立揭露，不單獨阻擋正式評估 | parser-v3 的三模型 parse rate 為 100%／80%／100%，已達預設門檻；台灣 base 四筆撞限均為未完成唯一答案的正常推理而非 serving/repetition 故障。這符合原規格「無法唯一解析算錯並另報 parse rate」，且避免依 validation 為單一模型提高推理預算 |
| 2026-07-22 | 使用者解鎖 Phase 4 正式評估 | 核准固定的 28,758 次生成與既有 256-token 協定；不擴及外部付費 API、adapter 發布或 Phase 5 |
| 2026-07-22 | 正式評估每 250 requests 封裝一個可驗證 Drive 分片 | 每片包含對齊的 public/private JSONL 與 immutable request plan；中斷後先核對 SHA-256、request IDs、model/suite 與 raw digest，再只補跑缺片，最多損失當下未完成的 250 次 |
| 2026-07-22 | Phase 4 能力結論以 strict accuracy 與 parse rate 並列 | adapter 的 MedQA／TMMLU+ 成績為 72.05%／61.53%，但 base 有大量 256-token hit；增益同時包含知識選答與輸出契約遵循，不把全數差距宣稱為純知識提升 |
| 2026-07-22 | catastrophic forgetting 依預註冊 non-inferiority 規則判定 | 五個控制科 adapter-base subject-macro 差異 +13.50 pp，paired bootstrap 95% CI [+10.41, +16.65] 高於 −2.0 pp 門檻，因此只在本次指定科目與協定下描述為未觀察到實質遺忘 |
| 2026-07-22 | 使用者確認進入 Phase 5，採筆電準備、4090 桌機驗收的移轉流程 | 目前筆電的 `nvidia-smi` 回報 RTX 2050 4GB，不能安全載入 12B；程式、測試與交付包先完成，真實 VRAM／TTFT／總延遲只在 RTX 4090 產生 |
| 2026-07-22 | Windows adapter 推論固定使用 Transformers + PEFT + bitsandbytes NF4 | Windows 不使用 vLLM；官方文件確認 bitsandbytes 提供 Windows wheel、PEFT 可載本機或 Hub adapter，並以 PyTorch 2.10/CUDA 12.8 相容線鎖定，不在本機硬編譯 |
| 2026-07-22 | Phase 5 不提供跳過 4090 硬體閘門的參數 | 防止在 4GB 過渡筆電誤下載並載入 12B；預檢需同時符合 Windows、RTX 4090、≥22 GiB、CC 8.9、CUDA 與 BF16 |
| 2026-07-22 | GGUF／Ollama 與 adapter 發布維持明確關閉 | GGUF 是不阻塞交付的 Colab Linux 選配；Hub 發布仍須指定 repo ID、visibility 並再次確認，不由 Phase 5 進入核准自動推送 |
| 2026-07-22 | Phase 5 大型產物以 no-overwrite `rename` 原子完成並有限重試 | 目的地依契約必須不存在；實機證實 Windows 對大型產物的 `Path.replace` 可回 `WinError 5`，同檔案改用 `Path.rename` 即成功。只重試權限錯誤 6 次、每次 0.25 秒，其他錯誤立即回報，耗盡後仍保留 `.partial` 供稽核 |
| 2026-07-22 | 離線 handoff 以新本機 root commit 接手 | 原 `.git` 未隨 handoff 移轉，receipt 將來源錨定在 `d97b183f...`；不偽造遺失的 parent history，以 commit `630bd5e` 建立可追蹤基準，模型權重與 secrets 仍排除 |
| 2026-07-22 | Windows inference 明確鎖定 PyTorch／torchvision CUDA 12.8 index | 一般 PyPI 會解析成 `torch 2.10.0+cpu`；依 uv 與 PyTorch 官方文件，Windows 的 torch 2.10／torchvision 0.25 改由 explicit `cu128` index 提供，Pillow 隨 inference group 鎖定，不編譯 wheel |
| 2026-07-22 | RTX 4090 acceptance 以內容安全 manifest 作正式證據 | 固定 synthetic probe 僅保存 prompt／raw output hash；驗證器重算硬體、base revision、adapter hash、量化、延遲、VRAM 與 strict answer，正文不進 Git |
| 2026-07-22 | 發布採 GitHub public、Hugging Face adapter private 的分階段策略 | 使用者核准 `kuotunyu/tw-med-llm-qlora` 公開與 `kuotunyu/tw-med-llm-qlora-adapter` 私人；先取得 hosted CI 與私人重載證據，未另行核准前不把 adapter 改為公開 |
| 2026-07-22 | 公開版清理時重新關閉 HF 發布 gate | 現有 credential 無法寫入原核准的 HF namespace；保留 GitHub 公開 URL，將 `adapter_repo_id` 清空並設 `publication.enabled=false`，待精確可寫入的私人目標再次確認 |

## 固定技術規格

- 主模型：`taide/Gemma-3-TAIDE-12b-Chat-2602`；BF16、VRAM ≥22 GiB。
- 12B 吞吐分級：≥70 GiB batch 8 / grad 2；≥38 GiB batch 4 / grad 4；≥22 GiB batch 1 / grad 16。
- T4 fallback：`twinkle-ai/gemma-3-4B-T1-it`；無 BF16、VRAM ≥14 GiB。
- 同尺寸基準：`google/gemma-3-12b-it` 或 `google/gemma-3-4b-it`。
- 資料：`bigbio/med_qa` converted-Parquet revision `e04abdc0672c54547fa1dbe36cfefc000e4f2657`。
- 評估：`ikala/tmmluplus` revision `81d53e38340c9ade988f7fed8996da6554b504f3`。
- QLoRA：4-bit、rank/alpha 16/16、effective batch 16、max sequence 2048、learning rate 5e-5、seed 3407。
- 清理優先序：test > validation > train；預期輸出 11,248 / 1,409 / 1,413。

## 成本與人工閘門

| 工作 | 估價方式 | 解鎖條件 |
| --- | --- | --- |
| 完整訓練 | A100 40GB 實際 session 1.143 小時；依 5.3 CU／小時約 6.06 CU，低於 calibration 的 26.49 CU 保守投影；未提供每 CU 單價，故不估金額 | 已完成並通過證據驗證 |
| Phase 4 雙軌評估 | 完整 28,758 requests 的完成 session 實測 2.542 小時；依 5.3 CU／小時估算 13.47 CU，低於 15.79 CU 投影與 18.95 CU 緩衝；先前校準／中斷 session 需另計 | 已完成並通過證據驗證 |
| 付費 LLM API | v1 不使用 | 若未來新增，先依 token 量與官方費率估價 |

## 產物連結

| 產物 | 狀態 | 位置 |
| --- | --- | --- |
| 專案設定 | 已建立 | [configs/project.toml](configs/project.toml) |
| 核心 MCQ 型別 | 已建立 | [src/tw_med_qlora/types.py](src/tw_med_qlora/types.py) |
| GPU 路由 | 已建立 | [src/tw_med_qlora/config.py](src/tw_med_qlora/config.py) |
| Project skill | 已建立並通過 validator | [.codex/skills/tw-med-qlora-workflow/SKILL.md](.codex/skills/tw-med-qlora-workflow/SKILL.md) |
| 五筆資料驗證報告 | 已完成 | [reports/data_sample_validation.json](reports/data_sample_validation.json) |
| 訓練 notebook | Phase 3 A100 40GB full mode 已完成並通過本機結構測試 | [notebooks/train_qlora.ipynb](notebooks/train_qlora.ipynb) |
| Phase 4 校準 notebook | parser-v3 已在 A100 完成；policy-v4 固定最終 token-hit 規則供重現 | [notebooks/evaluate_phase4.ipynb](notebooks/evaluate_phase4.ipynb) |
| Phase 4 正式評估 notebook | 已解鎖；A100、28,758 requests、step-700 adapter、固定 parser 與 Drive 分片續跑 | [notebooks/evaluate_phase4_full.ipynb](notebooks/evaluate_phase4_full.ipynb) |
| Phase 4 正式 manifest／receipt | 已驗證；A100 40GB、28,758 requests、123 shards、test-only 載入 | [reports/phase4/full/20260722T070936Z-run-manifest.json](reports/phase4/full/20260722T070936Z-run-manifest.json)／[receipt](reports/phase4/full/20260722T070936Z-receipt.json) |
| Phase 4 正式公開結果 | 已驗證；MedQA、TMMLU+、三 seed 穩定度與安全案例索引 | [reports/phase4/full/public/phase4-results.json](reports/phase4/full/public/phase4-results.json) |
| Phase 4 正式結果封存 | SHA-256 `1804c26d...2284f76`；不含題目、選項、prompt 或 raw output | [reports/phase4/full/20260722T070936Z-phase4-full-public.zip](reports/phase4/full/20260722T070936Z-phase4-full-public.zip) |
| Phase 4 正式本機驗證 | 28,758 unique pairs、suite counts、兩個 ZIP hash 與 10 私有案例均通過 | [reports/phase4/full/20260722T070936Z-validation.json](reports/phase4/full/20260722T070936Z-validation.json) |
| Phase 4 統計核心 | 單一字母／唯一 box 嚴格 parser、paired bootstrap、McNemar、遺忘判準已通過 CPU 測試 | [src/tw_med_qlora/evaluation.py](src/tw_med_qlora/evaluation.py) |
| Phase 4 正式分片核心 | deterministic request IDs、內容隔離、ZIP integrity、原子 Drive copy 與 resume 驗證已通過 CPU 測試 | [src/tw_med_qlora/phase4_full.py](src/tw_med_qlora/phase4_full.py) |
| Phase 4 正式證據驗證器 | 重算 request/suite 筆數、總表／分科、公開內容隔離、兩個 ZIP 與 receipt hash | [src/tw_med_qlora/cli/validate_phase4_full_evidence.py](src/tw_med_qlora/cli/validate_phase4_full_evidence.py) |
| Phase 5 Windows 推論 CLI | 已建立；單題、互動、固定 acceptance probe、adapter/base 核對、NF4/BF16/SDPA 與無法繞過的 4090 gate | [src/tw_med_qlora/local_inference.py](src/tw_med_qlora/local_inference.py) |
| Phase 5 4090 證據驗證器 | 已建立；驗證硬體、CUDA/BF16、模型 revision、adapter、量化、VRAM、TTFT、總延遲、可解析答案與隱私欄位 | [src/tw_med_qlora/cli/validate_phase5_evidence.py](src/tw_med_qlora/cli/validate_phase5_evidence.py) |
| Phase 5 移轉／驗收入口 | 已建立；先驗 Phase 3 full ZIP SHA-256、安全解出 step-700 adapter，再於桌機安裝官方 wheel、預檢與固定 probe | [scripts/run_phase5_acceptance.ps1](scripts/run_phase5_acceptance.ps1) |
| GitHub CPU CI | 已建立；Windows／Linux、Python 3.11、唯讀權限、無 secrets，執行 locked sync、Ruff、pytest 與 notebook freshness | [.github/workflows/ci.yml](.github/workflows/ci.yml) |
| Phase 5 完成度稽核器 | 已建立；逐項檢查 clean source、必要產物、step-700 adapter、4090 manifest、發布目標／origin 與 HF receipt，GGUF 明列為 optional | [src/tw_med_qlora/cli/phase5_status.py](src/tw_med_qlora/cli/phase5_status.py) |
| Phase 5 RTX 4090 acceptance | 已驗證；Windows、RTX 4090、PyTorch 2.10+cu128、NF4/BF16/SDPA、step-700 adapter、strict probe、VRAM 與延遲均通過 | [manifest](reports/phase5/20260722T131736Z-acceptance.json)／[validation](reports/phase5/20260722T131736Z-validation.json) |
| Adapter 模型卡 | 已完成草稿；包含基底條款、資料授權差異、研究用途、非醫療建議、正式結果與限制，發布目標仍保留變數 | [model_card/README.md](model_card/README.md) |
| Adapter 發布工具 | 已建立；預設 dry-run、檔案 allowlist、模型卡渲染與綁定 repo/visibility 的確認碼，外部寫入仍關閉 | [src/tw_med_qlora/cli/publish_adapter.py](src/tw_med_qlora/cli/publish_adapter.py) |
| 選配 GGUF／Ollama | 已建立雙重關閉的 A100 export notebook 與 Windows Ollama 驗收；不阻塞 adapter 交付 | [notebooks/export_gguf.ipynb](notebooks/export_gguf.ipynb) |
| Phase 4 首次 calibration manifest | 證據完整；結果因協定問題不作成績解讀 | [reports/phase4/calibration/20260722T052039Z-run-manifest.json](reports/phase4/calibration/20260722T052039Z-run-manifest.json) |
| Phase 4 首次 calibration 驗證摘要 | integrity pass；recalibration required | [reports/phase4/calibration/20260722T052039Z-validation.json](reports/phase4/calibration/20260722T052039Z-validation.json) |
| Phase 4 parser-v3 calibration manifest | A100 實跑完成，test 未載入 | [reports/phase4/calibration/20260722T061028Z-run-manifest.json](reports/phase4/calibration/20260722T061028Z-run-manifest.json) |
| Phase 4 parser-v3 calibration 摘要 | 20 題 validation／三模型／60 requests | [reports/phase4/calibration/20260722T061028Z-calibration-summary.json](reports/phase4/calibration/20260722T061028Z-calibration-summary.json) |
| Phase 4 parser-v3 本機驗證 | integrity pass；pass after protocol review | [reports/phase4/calibration/20260722T061028Z-validation.json](reports/phase4/calibration/20260722T061028Z-validation.json) |
| TMMLU+ 處理 | 固定抽樣、選項重排、私有 JSONL／安全 manifest 已通過 CPU 測試 | [src/tw_med_qlora/tmmlu.py](src/tw_med_qlora/tmmlu.py) |
| Checkpoint 封裝／恢復 | 已建立並通過 CPU 測試 | [src/tw_med_qlora/checkpointing.py](src/tw_med_qlora/checkpointing.py) |
| 全量資料 audit | 已完成 | [reports/data_validation.json](reports/data_validation.json) |
| Phase 2 run manifest | 已驗證 | [reports/phase2/20260721T160727Z-run-manifest.json](reports/phase2/20260721T160727Z-run-manifest.json) |
| Phase 2 Drive receipt | 已驗證 | [reports/phase2/20260721T160727Z-receipt.json](reports/phase2/20260721T160727Z-receipt.json) |
| Phase 2 驗證與成本摘要 | 已完成 | [reports/phase2/20260721T160727Z-validation.json](reports/phase2/20260721T160727Z-validation.json) |
| Phase 3 A100 manifest | 已驗證 | [reports/phase3/20260721T171557Z-run-manifest.json](reports/phase3/20260721T171557Z-run-manifest.json) |
| Phase 3 Drive receipt | 已驗證 | [reports/phase3/20260721T171557Z-receipt.json](reports/phase3/20260721T171557Z-receipt.json) |
| Phase 3 驗證與成本摘要 | 已完成 | [reports/phase3/20260721T171557Z-validation.json](reports/phase3/20260721T171557Z-validation.json) |
| Phase 3 full manifest | 已驗證 | [reports/phase3/full/20260722T014216Z-run-manifest.json](reports/phase3/full/20260722T014216Z-run-manifest.json) |
| Phase 3 full Drive receipt | 已驗證 | [reports/phase3/full/20260722T014216Z-receipt.json](reports/phase3/full/20260722T014216Z-receipt.json) |
| Phase 3 full trainer log | 已驗證 | [reports/phase3/full/20260722T014216Z-trainer_log.csv](reports/phase3/full/20260722T014216Z-trainer_log.csv) |
| Phase 3 full 訓練曲線 | 已驗證 | [reports/phase3/full/20260722T014216Z-training_curves.png](reports/phase3/full/20260722T014216Z-training_curves.png) |
| Phase 3 full 驗證摘要 | 13 項不變量通過 | [reports/phase3/full/20260722T014216Z-validation.json](reports/phase3/full/20260722T014216Z-validation.json) |
| 評估結果 | 正式 MedQA／TMMLU+ 結果與統計均已完成 | [reports/phase4/full/public/phase4-results.json](reports/phase4/full/public/phase4-results.json) |
| Adapter | step 700 已完成並通過正式評估；私人發布 namespace 待重新確認，尚未上傳 | `adapter_repo_id` 暫空 |

## 執行紀錄

| 日期 | Phase | 動作 | 證據／結果 | 下一步 |
| --- | --- | --- | --- | --- |
| 2026-07-21 | 規劃 | 驗證環境、模型存取、資料 schema 與現行官方 API | RTX 2050 4GB；TAIDE gated access 可用；MedQA 原始切分 11,298/1,412/1,413 | 建立 Phase 0 |
| 2026-07-21 | 0 | 建立骨架、設定、README、核心型別與 workflow skill | `uv lock` 與 `uv sync` 使用 CPython 3.11.15；`uv.lock` 已建立 | 執行品質檢查 |
| 2026-07-21 | 0 | 執行 Phase 0 品質檢查 | `uv run pytest`：10 passed；`uv run ruff check .`：passed；skill validator：valid | 檢查 Git 與敏感檔 |
| 2026-07-21 | 0 | 初始化 `main` Git repo 並驗證 ignore | `git check-ignore -v .env` 命中 `.gitignore:1`；`.env` 未出現在 `git status --short` | 等待作者身分與 Phase 0 確認 |
| 2026-07-21 | 0 | 使用者核准 Phase 0 | 已收到進入下一步的明確指示；公開 Git 作者身分仍待設定 | 取得 repo-local Git 作者身分並 commit |
| 2026-07-21 | 0 | 設定 repo-local Git 作者身分 | 使用者指定 Git 作者名稱 `kuotunyu`；不更動 global config | 完成 Phase 0 commit |
| 2026-07-21 | 0 | 完成 Phase 0 commit | `9d90459 chore: initialize research workflow`；worktree clean | 進入已核准 Phase 1 |
| 2026-07-21 | 1 | 查核資料與 chat-template API | Context7 不可用；改查官方 `snapshot_download`、Parquet `iter_batches`、`apply_chat_template` 文件 | 建立五筆 gate |
| 2026-07-21 | 1 | 執行五筆資料 gate | 5/5 schema、UTF-8、A–D、答案文字、chat-template round trip 與單一 BOS 全數通過；報告不含題目正文 | 提交 gate 後才進行全量處理 |
| 2026-07-21 | 1 | 診斷全量品質規則 | train 有 3 筆缺值、5 筆歧義選項；validation 有 1 筆歧義選項；NFKC 會過度合併 | 採保守 whitespace-collapse + casefold |
| 2026-07-21 | 1 | 執行全量處理與隔離驗證 | 原始 11,298/1,412/1,413；品質排除 9、重複排除 44；輸出 11,248/1,409/1,413；跨 split overlap 0 | 驗證重跑 hash |
| 2026-07-21 | 1 | 連續重跑全量處理 | report、train、validation、test SHA-256 全數一致；report hash `4823283977f2064f859f923056f13f81dd6663704fddd3b41086e8ca4587c5ed` | 測試、提交並展示 Phase 1 |
| 2026-07-21 | 1 | 執行 Phase boundary 驗收 | `pytest` 23 passed；Ruff、uv lock、skill validator 通過；raw/processed/`.env` 均確認 ignored | 提交並等待 Phase 2 核准 |
| 2026-07-21 | 1→2 | 使用者確認進入 Phase 2 | Phase 1 正式完成；Phase 2 僅先建立 Colab notebook 與 CPU 可驗證項目 | 查核現行 Unsloth/TRL API |
| 2026-07-21 | 2 | 查核 Gemma 3 訓練 API 與套件版本 | Context7 本工作階段不可用；官方文件確認 `FastModel`、4-bit、vision layers off、T4 FP16 保護；鎖定 `unsloth/unsloth-zoo 2026.7.4` 與 TRL 0.22.2 介面 | 建立 100 筆／10 steps notebook |
| 2026-07-21 | 2 | 建立可重建 Colab smoke notebook | 自動 GPU 分流、HF Secret、Drive、MedQA hash/隔離、response-only、10-step hard gate、adapter 重載、manifest、成本輸出均已加入 | 執行本機結構驗收 |
| 2026-07-21 | 2 | 執行 notebook 本機驗收 | `uv sync --all-groups --link-mode copy` 通過；`pytest` 45 passed；Ruff 通過；notebook builder `--check` 通過；未在 4GB GPU 載入模型 | 使用者在 Colab 執行 smoke |
| 2026-07-21 | 2 | 加入 premium GPU 吞吐分級 | 使用者確認 Colab Pro 可積極使用較佳 GPU；新增 80GB/40GB/24GB/16GB 四級設定，全部維持 effective batch 16 | 重跑 notebook 結構驗收 |
| 2026-07-21 | 2 | 驗收 premium GPU 路由 | `pytest` 48 passed；Ruff、uv lock、notebook builder `--check`、skill validator 全數通過 | 使用者在可用時選 premium GPU 執行 smoke |
| 2026-07-21 | 2 | 診斷首次 L4 smoke 模型載入失敗 | L4 22.034 GiB、BF16 與 12B 路由正確；Unsloth 網路錯誤後離線重試，在部分 HF cache 上使 Transformers 4.56.2 收到 `checkpoint_files=[None]` | 改為載入前完整快取與 shard 驗證 |
| 2026-07-21 | 2 | 修正模型 snapshot 載入流程 | 固定 revision 先以 Hub `snapshot_download` 完整下載，核對遠端檔案大小、index 與每個 shard，再以 `local_files_only=True`、`use_safetensors=True` 交給 Unsloth；移除舊 `hf-transfer` 開關 | 重建 notebook 並執行本機驗收 |
| 2026-07-21 | 2 | 驗收修正版 notebook | 主模型 revision 對應 6 個 shards / 25.376 GiB，fallback 對應 2 個 shards / 8.478 GiB，兩者皆有 index；`pytest` 48 passed，Ruff、uv lock、builder `--check`、skill validator 與 ignore 檢查全數通過 | 以全新／重啟的 Colab session 重跑 smoke |
| 2026-07-21 | 2 | 診斷第二次 L4 smoke tokenizer 載入失敗 | 17 個 snapshot 檔於 94 秒完成，6/6 shards 載入成功；Unsloth 的 VLM tokenizer fallback 未沿用 pinned revision，離線時以空的 `tokenizer_name` 查找而失敗 | 明確指定已驗證的本機 snapshot 作為 tokenizer/processor 來源 |
| 2026-07-21 | 2 | 修正 tokenizer/processor 解析 | 載入前驗證 tokenizer；VLM 額外驗證 processor/preprocessor，並把 `tokenizer_name=str(snapshot_path)` 傳給 Unsloth，模型仍保留正式 Hub ID 與 revision | 重建 notebook 並執行本機驗收 |
| 2026-07-21 | 2 | 驗收第二版修正 | PyPI 官方 Unsloth 2026.7.4 wheel（SHA-256 `843a217a...39c`）確認 `tokenizer_name` override 會由 resolver 傳入 processor；`pytest` 48 passed，Ruff、uv lock、builder `--check`、skill validator 與 ignore 檢查全數通過 | 在保留完整 HF cache 的 Colab session 重新執行模型載入格 |
| 2026-07-21 | 2 | 診斷 response-only masking 失敗 | 模型、tokenizer、100 筆 chat rendering 與 SFTTrainer 建立均成功；TRL 因 `processing_class` 是 VLM processor 而選到不支援事後 masking 的 vision collator | 純文字訓練改用 processor 內層 tokenizer |
| 2026-07-21 | 2 | 修正純文字 collator 與 masking | 官方 Gemma 3 vision notebook亦以內層 tokenizer 作 `processing_class`；Transformers 4.56.2 確認 `pixel_values`/`token_type_ids` 可為空。改用 language-model collator，保留 Unsloth response-only masking並新增首筆 labels 稽核 | 重建 notebook 並執行本機驗收 |
| 2026-07-21 | 2 | 驗收純文字 response-only 修正 | TRL 0.22.2 原始碼確認 inner tokenizer 走 `DataCollatorForLanguageModeling`；notebook 會拒絕 vision collator並輸出 `MASKING_AUDIT`。`pytest` 48 passed，Ruff、uv lock、builder `--check`、skill validator 與 ignore 檢查全數通過 | 在保留模型 cache 的 Colab session 重跑 smoke |
| 2026-07-21 | 2 | 診斷 adapter 重載失敗 | smoke 已通過資料、模型、masking、10-step 訓練與 adapter 保存，卡在最後重載；Unsloth PEFT 自動路徑離線時未把 pinned revision 傳給 base `AutoConfig`，因此錯查未快取的 `main` | 將 pinned base 與 adapter 的重載步驟拆開 |
| 2026-07-21 | 2 | 修正 pinned base + PEFT adapter 重載 | 依 PEFT 0.19.1 官方 `PeftModel.from_pretrained(model, model_id, is_trainable=False)`：先從已驗證 snapshot 重建 4-bit base，再掛載本機 adapter；保留並驗證公開用 Hub base ID | 重建 notebook 並執行本機驗收 |
| 2026-07-21 | 2 | 驗收 adapter 重載修正 | 重載 cell 明確稽核 default adapter、Hub base ID、LoRA 參數存在且 inference mode 為 frozen；`pytest` 48 passed，Ruff、uv lock、builder `--check`、skill validator 與 ignore 檢查全數通過 | 在保留模型 cache 的 Colab session 重跑 smoke，取得最終 manifest |
| 2026-07-21 | 2 | 診斷 validation generation 格式錯誤 | pinned base 與 PEFT adapter 重載皆成功；最後呼叫 VLM processor 的 `apply_chat_template` 時，純字串 content 被當成多模態項目清單而失敗 | generation 改用內層文字 tokenizer |
| 2026-07-21 | 2 | 修正重載後純文字 generation | Transformers 4.56.2 `PreTrainedTokenizerBase.apply_chat_template` 明確接受 `list[dict[str, str]]`；改以 `reload_text_tokenizer` 套 TAIDE 字串 chat template，processor 不參與純文字 prompt | 重建 notebook 並執行本機驗收 |
| 2026-07-21 | 2 | 驗收 validation generation 修正 | 重載 cell 可重跑並清理殘留 GPU 物件；靜態檢查禁止對 VLM processor 建立純文字 generation inputs。`pytest` 48 passed，Ruff、uv lock、builder `--check`、skill validator 與 ignore 檢查全數通過 | 在保留模型 cache 的 Colab session 重跑，取得 strict A–D 與 manifest |
| 2026-07-22 | 2 | 強化全新 Colab 啟動流程 | 使用者已關閉舊 runtime；安裝格後新增 module/版本 gate，若 Unsloth 未安裝會在第 1 節停止，不再延遲到模型載入格 | 重建 notebook 並執行本機驗收 |
| 2026-07-22 | 2 | 驗收全新 runtime notebook | 依賴 gate 不預先 import Unsloth，只以 module spec 與 distribution metadata 驗證 9 個關鍵套件；`pytest` 48 passed，Ruff、uv lock、builder `--check`、skill validator 與 ignore 檢查全數通過 | 使用者建立全新 Colab GPU runtime 並從最上方全部執行 |
| 2026-07-22 | 2 | 完成 L4 smoke test | 100 筆／10 steps；無 OOM 或非有限 loss；峰值 allocated/reserved 9.33/11.19 GiB；adapter 保存、重載及 validation 嚴格 A–D 解析通過；test 未進 trainer | 下載 manifest 與 receipt |
| 2026-07-22 | 2 | 驗證並歸檔 smoke 證據 | 原始 JSON 未含 secret、題目正文或完整生成內容；新增自動驗證器與 5 項 gate 測試；完整 703 steps 依 L4 實測投影 13.36 小時、20.58 CU | 執行 Phase boundary 驗收並等待使用者確認 |
| 2026-07-22 | 2 | 執行 Phase boundary 驗收 | `pytest` 53 passed；Ruff、uv lock、notebook builder `--check`、skill validator、敏感內容掃描與 ignore 檢查全數通過 | 停在 Phase 2 閘門，等待使用者確認進入 Phase 3 |
| 2026-07-22 | 2→3 | 使用者確認進入 Phase 3 | 已取得明確核准，GPU 策略為優先使用 A100；完整長跑仍須先依 A100 calibration 更新 CU 估算 | 查核 checkpoint／resume 官方 API |
| 2026-07-22 | 3 | 查核 checkpoint 與 validation API | Context7 不可用；固定版本官方原始碼確認 `eval_dataset`、`eval_strategy="steps"`、`on_save` 與 `resume_from_checkpoint=<path>`；可續訓 checkpoint 必須保留 optimizer、scheduler、RNG | 建立完整性封裝與恢復工具 |
| 2026-07-22 | 3 | 建立 checkpoint 原子封裝與恢復核心 | checkpoint 先在本機封裝，複製後核對 SHA-256，再以 partial rename 發布到 Drive；latest metadata 綁定實驗 fingerprint，恢復時驗證大小、hash、member 與必要續訓狀態；保留最近兩份 | 將 callback 與 A100 calibration 接入 notebook |
| 2026-07-22 | 3 | 建立 A100 calibration／full training notebook | 預設以 100 train／10 steps 加 100 validation 分別校準訓練、checkpoint 與評估時間；premium profile gate、CU 必填、full 三重閘門、同 hardware profile 約束、11,248/1,409 train/validation、每 100 steps eval/save、Drive callback、自動 resume、曲線、manifest 與模型卡草稿均已接入；不含 Hub push | 執行 notebook 結構與全套測試 |
| 2026-07-22 | 3 | 驗收 Phase 3 calibration notebook | `pytest` 60 passed；Ruff、uv lock、notebook builder `--check`、skill validator 與 ignore 檢查通過；所有 code cell 可編譯且無保存輸出；checkpoint 封裝、損毀拒絕、fingerprint mismatch、保留兩份及恢復 round trip 均有 CPU 測試 | 交付 notebook，等待 A100 calibration 證據 |
| 2026-07-22 | 3 | 完成 A100 40GB calibration | 100 train／100 validation／10 steps；20.53 秒／step，checkpoint 12.13 秒，100 筆 validation 30.85 秒；無 OOM/NaN/Inf，峰值 reserved 12.36 GiB，adapter 重載與 checkpoint step 10 恢復通過，test 未進 trainer | 下載並驗證 manifest／receipt |
| 2026-07-22 | 3 | 驗證並歸檔 A100 calibration 證據 | 原始 JSON 不含 secret、題目正文或完整生成；以資源面板實際 5.3 CU／小時修正示例輸入，完整訓練投影 5.00 小時／26.49 CU，含 20% 緩衝 31.78 CU | 執行完整驗收，停在成本閘門等待使用者核准 |
| 2026-07-22 | 3 | 執行 calibration 證據完整驗收 | `pytest` 67 passed；Ruff、uv lock、notebook builder `--check`、skill validator、敏感內容與 ignore 檢查通過；驗證器另覆蓋 OOM、test 洩漏、checkpoint 缺件、恢復失敗與 fingerprint mismatch 拒絕 | 等待使用者明確核准完整長跑 |
| 2026-07-22 | 3 | 使用者核准完整長跑 | 使用者接受 A100、1 epoch 與 CU 消耗，預定隔日啟動；本次核准僅涵蓋 Phase 3，不自動進入 Phase 4 | 產生無需手動改碼的 full notebook |
| 2026-07-22 | 3 | 產生已核准的 full notebook 與事後驗證器 | 固定 `primary_40g`、703 steps、每 100 steps eval/save、自動 resume、5.3 CU／小時與校準時間；小型 log／曲線／模型卡／套件清單獨立同步 Drive 並寫入 receipt；驗證器預先檢查完整資料筆數、test 隔離、checkpoint step 700 恢復與證據 hash | 執行本機全套驗收並交付 |
| 2026-07-22 | 3 | 驗收已核准的 full notebook | `pytest` 75 passed；Ruff、uv lock、notebook builder `--check` 與 skill validator 通過；full 驗證器另覆蓋未核准、test 洩漏、step 不足、恢復點錯誤、fingerprint 與曲線 hash mismatch | 提交並交付隔日執行 |
| 2026-07-22 | 3 | 完成 A100 40GB full training | 11,248 train／1,409 validation、1 epoch、703 steps；平均 training loss 0.287178，step 700 validation loss 0.227021；訓練 wall time 1.11 小時，session 約 1.143 小時／6.06 CU | 稽核額外 step 703 eval NaN 並執行 recovery |
| 2026-07-22 | 3 | 從已驗證 checkpoint recovery | 額外 step 703 validation 為 NaN，但 65,470,464 個 LoRA 參數皆有限；選用 step 700，Drive archive SHA-256、checkpoint step、PEFT key set 與轉為目標 BF16 後逐 tensor exact match 均通過 | 保存、重載 adapter 並同步證據 |
| 2026-07-22 | 3 | 驗證並歸檔 full run 證據 | adapter 重載與 strict A–D probe 通過；manifest、receipt、trainer log、曲線 SHA-256 相符，本機驗證器 13 項不變量通過；單題 probe 僅驗證格式，不作能力宣稱 | 執行 Phase boundary 全套驗收，停在 Phase 3 等待確認 |
| 2026-07-22 | 3 | 執行 Phase 3 boundary 驗收 | `pytest` 78 passed；Ruff、uv lock、notebook builder `--check`、project skill validator、ignore 與 diff checks 全數通過；README 僅記錄實測訓練結果，MedQA test／TMMLU+ 仍標為尚未執行 | 提交 Phase 3 證據，等待使用者確認進入 Phase 4 |
| 2026-07-22 | 4 | 使用者確認進入 Phase 4 | 核准開始雙軌評估實作；大量 test 生成仍受 20 題 calibration 成本閘門限制 | 查核官方 API、科目筆數與 request contract |
| 2026-07-22 | 4 | 查核 Twinkle Eval／vLLM 與 TMMLU+ | Context7 工具仍未配置，依規範改查官方文件與原始碼；鎖定 Twinkle Eval 2.8.0／commit `470bbec...`、vLLM 0.25.1；13 科 test 合計 5,573 題，validation 校準不載入 test | 建立評估核心與校準 notebook |
| 2026-07-22 | 4 | 建立統計、TMMLU+ 與 serving 安全核心 | 嚴格 parse、內容安全 prediction、總表／分科、paired bootstrap、McNemar、−2 pp non-inferiority、固定抽樣／重排、adapter ZIP 驗證、vLLM CLI 與 28,758 request contract 均有 CPU 測試 | 產生 20 題 A100 calibration notebook |
| 2026-07-22 | 4 | 驗證固定 revision 的 13 科 CSV | validation 合計 619、test 合計 5,573；法律原理 test 含相同內容列，因此 ID 納入固定 source row ordinal，保留官方 106 列且避免 paired result 碰撞 | 補上重複列測試並重建 notebook |
| 2026-07-22 | 4 | 產生 validation-only A100 calibration notebook | 三模型共用 20 題、60 次生成；只下載 `*_val.csv`，adapter ZIP 先驗 hash/base，raw output 私存 Drive，公開 manifest 無題目正文；full gate 固定關閉 | 執行完整本機驗收並交付使用者 |
| 2026-07-22 | 4 | 完成 calibration notebook 本機驗收 | `pytest` 105 passed；Ruff、uv lock、train/eval notebook builder `--check`、project skill validator、diff 與 ignore checks 全數通過；兩份 notebook 無保存輸出，完整 test gate 仍關閉 | 提交 Phase 4 校準產物，請使用者在 A100 執行 |
| 2026-07-22 | 4 | 診斷首次 calibration 啟動失敗 | vLLM 0.25.1 plain PyPI wheel 載入 `_C_stable_libtorch` 時要求缺少的 `libcudart.so.13`；server 尚未啟動，舊 `stop_server` 又對已退出 PID 發送 signal，造成第二個 `ProcessLookupError`；未下載或讀取 test | 改用官方 CUDA 12.9 wheel 並讓清理函式冪等 |
| 2026-07-22 | 4 | 產生 CUDA 12.9 修正版 calibration notebook | Context7 未配置，依官方 vLLM 安裝文件與 GitHub release API 固定 `0.25.1+cu129` wheel、uv 0.11.31、wheel SHA-256；模型下載前以獨立程序驗證 vLLM 原生匯入、PyTorch CUDA 12.9 與 GPU；`stop_server` 可安全處理已退出程序 | 完成本機全套驗收後，交付全新 A100 runtime 重跑 |
| 2026-07-22 | 4 | 驗收 CUDA 12.9 修正版 notebook | Linux/Python 3.12 跨平台 uv dry-run 成功解析 216 個套件，包含 `torch 2.11.0+cu129`、`torchvision 0.26.0+cu129` 與官方 vLLM wheel；`pytest` 105 passed，Ruff、uv lock、兩份 notebook builder、skill validator、無保存輸出與 ignore checks 全數通過 | 提交修正版，請使用者刪除舊 runtime 後以 A100 全部執行 |
| 2026-07-22 | 4 | 驗證首次成功的 20 題 calibration 證據 | A100／CUDA 12.9／vLLM 0.25.1+cu129、60 requests、兩次 server 啟動與私人 ZIP SHA-256 均通過；test_files_loaded=0、full gate=false；原廠 17/20 boxed，adapter 20/20 單一字母，台灣 base 20/20 在 32-token 內未形成唯一答案 | 不採用該次 accuracy／成本作正式決策，修正答案協定 |
| 2026-07-22 | 4 | 產生 parser-v3 calibration notebook | 嚴格接受單一 A–D 或唯一簡單 boxed A–D；max tokens 256；各模型 parse rate ≥80% 且零 token-limit hit，否則安全封存證據並維持 full gate 鎖定；新增私有證據驗證器與安全格式稽核 | 完成本機全套驗收後，交付 A100 重跑 |
| 2026-07-22 | 4 | 驗收 parser-v3 與首次 calibration 證據 | `pytest` 116 passed；Ruff、uv lock、train/eval notebook builder `--check`、project skill validator、私有 ZIP／`.env` ignore 與 diff checks 全數通過；證據驗證結果為 integrity pass／recalibration required，test 仍未載入 | 提交修正版並交付使用者以全新 A100 runtime 重跑 |
| 2026-07-22 | 4 | 完成 parser-v3 A100 calibration | 20 題 validation／60 requests；原廠、台灣 base、adapter accuracy 50%／45%／55%，parse rate 100%／80%／100%；台灣 base 有 4 筆撞 256-token 上限；投影完整評估 2.98 小時／15.79 CU，含緩衝 18.95 CU | 下載並驗證四項證據，完整 test 維持鎖定 |
| 2026-07-22 | 4 | 驗證 parser-v3 證據並審查 token hit | manifest、receipt、summary、私人 ZIP SHA-256 與 60 筆公開計數均重算一致；test_files_loaded=0、full gate=false；四筆撞限為未完成唯一答案的正常推理，無重複生成。依 strict parser 計錯並另報 parse rate，校準狀態 `pass_after_protocol_review` | 完成全套本機驗收，展示成本並等待使用者明確解鎖正式評估 |
| 2026-07-22 | 4 | 完成正式評估成本閘門前驗收 | `pytest` 118 passed；Ruff、uv lock、train/eval notebook builders、project skill validator、私人 ZIP／`.env` ignore 與 diff checks 全數通過；公開報告無題目或 raw output | 提交校準證據與最終協定，等待使用者核准 28,758-request 正式評估 |
| 2026-07-22 | 4 | 使用者明確解鎖正式評估 | 核准語句「確認解鎖 Phase 4 正式評估」已寫入設定；範圍固定為 28,758 次 A100 本機生成，不包含外部付費 API、HF 發布或 Phase 5 | 重新核對官方 serving／評估介面並實作正式 notebook |
| 2026-07-22 | 4 | 重新核對 vLLM／Twinkle Eval 官方介面 | Context7 仍未配置，依規範查閱官方文件；確認 OpenAI-compatible chat server、啟動時 LoRA 掛載、以 model 名稱選 adapter，以及 Twinkle `box`／exact-match／resume 能力 | 沿用已校準 prompt/parser，加入正式分片與續跑 |
| 2026-07-22 | 4 | 建立可續跑正式評估 notebook 與分片核心 | 固定 A100 40GB、CUDA 12.9、step-700 adapter、test-only 資料載入、三模型與三 seeds；每 250 requests 封裝 public/private ZIP，原子同步 Drive，重跑驗證完整分片後略過；公開彙總含 MedQA paired CI/McNemar、TMMLU+ 分科／穩定度與 −2 pp non-inferiority，完整 raw 僅留私人分片 | 執行全套本機驗收並提交，交付使用者 A100 執行 |
| 2026-07-22 | 4 | 完成正式評估 notebook 本機驗收 | `pytest` 130 passed；Ruff、uv lock 與 train/calibration/full notebook builders `--check` 通過；正式 notebook 無保存輸出，核准 request contract 恰為 28,758，無 Hub push 或外部付費 API；事後驗證器可重算公開結果與封存 hash | Conventional Commit 後請使用者上傳正式 notebook、選 A100 並全部執行 |
| 2026-07-22 | 4 | 完成 A100 正式雙軌評估 | 28,758 requests／123 shards 全數完成；MedQA 原廠、base、adapter accuracy 66.17%／56.40%／72.05%；TMMLU+ 全體 53.47%／46.80%／61.53%；完成 session 2.542 小時／估算 13.47 CU | 下載四項最終證據並執行本機重算 |
| 2026-07-22 | 4 | 驗證正式評估與隱私隔離 | 28,758 unique model/request pairs、所有 suite counts、公開／私人 ZIP SHA-256 與 10 個代表案例全部一致；公開 archive 禁止欄位 `question`／`choices`／`prompt`／`raw_output` 均未洩漏 | 匯入公開摘要、撰寫 README 與私有質性分析 |
| 2026-07-22 | 4 | 完成正式結果解讀 | MedQA adapter-base +15.64 pp，bootstrap 95% CI [13.38, 17.98]、McNemar p=6.11e-40；控制科 +13.50 pp，CI [+10.41, +16.65]，通過 −2 pp non-inferiority；另揭露 base 低 parse rate 與題庫缺圖／承上題限制 | 執行 Phase 4 boundary 全套驗收並停在使用者確認閘門 |
| 2026-07-22 | 4 | 完成 Phase 4 boundary 驗收 | `pytest` 130 passed；Ruff、uv lock、三份 notebook builders、正式證據驗證器、diff／ignore checks 全數通過；清理可重建的舊 package metadata 後一般 `uv run` 同步恢復正常 | 建立結果 commit，等待使用者明確確認進入 Phase 5 |
| 2026-07-22 | 5 | 使用者明確確認進入 Phase 5 | 核准 Windows 本機推論與發布準備；4090 位於另一台桌機，過渡筆電先完成所有 CPU 可驗證項目 | 查核現行官方 API 並建立可移轉驗收流程 |
| 2026-07-22 | 5 | 查核 Windows CUDA／Gemma 3／PEFT／GGUF 官方介面 | Context7 仍未配置；官方資料確認 TAIDE 模型為 gated、授權標籤 `gemma-version-taide-models-license-agreement`、文字輸入建議、PEFT frozen adapter 載入、Windows bitsandbytes wheel 與 Unsloth `save_pretrained_gguf(..., q4_k_m)` | 實作本機推論與發布前安全閘門 |
| 2026-07-22 | 5 | 建立 4090 推論與證據核心 | `tw-med-local-infer` 支援 `--prompt`、`--interactive`、`--acceptance`、`--preflight-only`；在模型下載前拒絕非核准硬體與 adapter/base mismatch，manifest 不保存 prompt/raw output；20 項聚焦測試通過 | 補齊 4090 移轉腳本、模型卡與發布 gate |
| 2026-07-22 | 5 | 建立可攜式 adapter 移轉與一鍵驗收 | 移轉器先核對 Phase 3 archive SHA-256 `2c537d...47e43e`、拒絕 ZIP slip／覆寫並只抽出 `adapter/`；PowerShell 依序執行鎖定 wheel、4090 預檢與固定 A–D probe；全套 `pytest` 154 passed、Ruff／uv lock／三 notebook builders／PowerShell syntax 均通過 | Conventional Commit 後建立模型卡與發布前驗證 |
| 2026-07-22 | 5 | 建立模型卡與安全發布計畫 | 模型卡揭露基底 gated 條款、資料授權差異、step-700 選擇、正式評估與限制；發布 CLI 預設只做 dry-run，僅 allowlist adapter/tokenizer/渲染模型卡，且必須以 4090 acceptance、repo 設定與綁定確認碼解鎖 | 建立選配 GGUF／Ollama 與完整桌機交接文件 |
| 2026-07-22 | 5 | 建立選配 GGUF／Ollama 與 4090 交接文件 | A100 notebook 以 repo gate 加 notebook gate 雙重關閉，只允許 `q4_k_m`、Drive 輸出且不 push；Windows 以 Ollama 匯入合併 GGUF並做固定 probe；交接清單列出 Phase 3 ZIP hash、USB 排除項目與唯一驗收入口 | 重建受設定影響的 notebook 並執行 Phase 5 全套本機驗收 |
| 2026-07-22 | 5 | 完成發布／選配 export 本機驗收 | `pytest` 162 passed；Ruff、uv lock、四份 notebook builders、project skill validator、兩份 PowerShell syntax、HF ModelCard parser、credential／ignore／diff checks 全數通過；模型卡與外部發布仍未填入虛構目標 | 建立聚焦 commit，等待 4090 acceptance 與使用者指定發布目標 |
| 2026-07-22 | 5 | 建立可驗證的 USB handoff bundle | bundle 只接受 clean Git snapshot，排除 `.env`／`.venv`／私有輸出，核對 adapter/base 並限制檔案 allowlist；ZIP 內外 receipt 記錄 commit、逐檔 hash 與最終 archive hash；`pytest` 167 passed、Ruff 通過 | 完成 Phase 5 交付前全套檢查並建立聚焦 commit |
| 2026-07-22 | 5 | 提交 USB handoff bundle | commit `ce28211 feat: add verified 4090 handoff bundle`；worktree 乾淨，尚未產生沒有真實 adapter 的虛構 bundle | 建立公開 repo 的 CPU CI |
| 2026-07-22 | 5 | 建立公開 repo 的 CPU CI | Context7 未配置，依官方 GitHub workflow 與 uv integration 文件建立 Windows／Linux matrix；workflow 僅有 `contents: read`，無 secrets／發布權限，固定 `setup-uv v8.1.0` action SHA 與 uv 0.11.30，使用 `uv sync --locked` | 以隔離 Windows 環境模擬完整 CI 後提交 |
| 2026-07-22 | 5 | 完成隔離 Windows CI 模擬 | 全新 dev-only virtualenv 安裝 19 個套件，Ruff、pytest 170 passed、四份 notebook freshness 全數通過；暫存環境不含 data／inference 依賴。Linux matrix 須等建立 GitHub remote 後由 hosted runner 提供最終證據 | 增加 workflow 安全契約測試並建立聚焦 commit |
| 2026-07-22 | 5 | 完成 CI 安全契約與全套本機驗收 | workflow 測試確認唯讀權限、無 secrets／upload、固定 action SHA、Windows／Linux matrix、locked install 與四份 builder；全套 `pytest` 173 passed、Ruff 通過 | 建立 CI commit，等待外部 adapter／4090／repo 目標 |
| 2026-07-22 | 5 | 提交跨平台 CPU CI | commit `0915334 ci: add locked cross-platform validation`；Windows 隔離模擬已通過，worktree 乾淨；未建立 remote、未觸發 hosted runner、未賦予寫入權限 | 等待 Phase 3 adapter ZIP；完成真實 USB bundle 與 4090 acceptance |
| 2026-07-22 | 5 | 建立 Phase 5 完成度稽核器 | 將 clean worktree、必要產物、step-700 adapter、4090 acceptance、模型卡、HF/GitHub 目標、origin、顯式發布 gate 與 publication receipt 固定成 pass/pending/fail 契約；只有全部必需證據通過才回報完成，GGUF 明列 optional | 執行完整測試與真實目前狀態快照 |
| 2026-07-22 | 5 | 驗收完成度稽核器 | 6 項聚焦測試涵蓋 pending、handoff ready、dirty tree、receipt 安全與完整完成路徑；全套 `pytest` 179 passed、Ruff 通過 | 建立聚焦 commit，於 clean commit 上執行目前狀態稽核 |
| 2026-07-22 | 5 | 診斷並恢復本機一般 CLI 啟動 | 既有 `.venv` 的 editable metadata 曾遺失，`pytest` 因 `pythonpath=src` 未暴露；一般 `python -m` 實跑正確失敗。舊 uv hardlink dist-info 在 Windows 無法直接移除，將該生成目錄保留為 `.pre-sync` 後以 locked/inexact copy 成功重裝；fresh CI 環境原已通過，4090 handoff 明確不攜帶舊 `.venv` | 在 clean commit 重跑 Phase 5 status CLI |
| 2026-07-22 | 5 | 產生 pre-4090 readiness 快照 | commit `6cd1fef` 上 source、README、模型卡、推論腳本、acceptance 腳本與 CI 全部 pass；step-700 adapter、RTX 4090、發布目標／gate／origin／HF receipt 全部 pending；`ready_for_handoff=false`、`phase5_complete=false`，符合真實狀態 | 提交快照與紀錄，等待外部 adapter ZIP |
| 2026-07-22 | 5 | 驗證並追蹤 pre-4090 readiness | 新增契約測試確認快照未宣稱 handoff／publication／Phase 5 完成、外部五項 gate 維持 pending，且不含 `HF_TOKEN` 或本機絕對路徑；全套 `pytest` 180 passed、Ruff 通過 | 建立 docs commit，等待使用者放入 Phase 3 full ZIP |
| 2026-07-22 | 5 | 驗證 Phase 3 full ZIP 並修正 Windows 原子完成競態 | 使用者下載的 ZIP 為 113,079,186 bytes，SHA-256 `2c537dfd...47e43e`，與正式 receipt 完全一致；step-700 adapter 的 base ID／revision 與逐檔 hash 已通過。大型目錄與 handoff ZIP 在 `Path.replace` 完成時各遇一次 `WinError 5`；實機最小診斷確認 `Path.rename` 立即成功，符合既有 no-overwrite 契約 | 改用 no-overwrite rename、有限重試並重新驗收 handoff |
| 2026-07-22 | 5 | 驗收 Windows 原子完成修正 | 全套 `pytest` 183 passed、Ruff 與 project skill validator 通過；Phase 3 ZIP、adapter 權重與診斷 partial 均由 `artifacts/` ignore 規則隔離，Git diff check 通過 | 建立聚焦 Conventional Commit 後產生正式 handoff |
| 2026-07-22 | 5 | 實跑並獨立驗證離線 handoff | builder 以 clean commit 封裝已驗證 adapter；重新計算 archive 大小／SHA-256，執行 ZIP CRC，核對 ZIP 內 receipt 與 sidecar，並確認不含 secrets、private generations 或環境；`phase5_status --adapter ...` 回報 `ready_for_handoff=true` | 以最新文件 commit 重建最終 handoff，交付 RTX 4090 桌機 |
| 2026-07-22 | 5 | 接手離線 handoff 並恢復本機版本控制 | 129 個 source 中 128 個位元級相符、兩份來源規範文字同步；12 個 adapter 檔案全部雜湊相符。原 Git history 未包含在 bundle，以來源 commit `d97b183f...` 為錨點建立 root commit `630bd5e`，secrets／weights／outputs 均未追蹤 | 在目前 RTX 4090 執行 locked acceptance |
| 2026-07-22 | 5 | 診斷首次 4090 preflight 失敗 | 一般 PyPI 解析成 `torch 2.10.0+cpu`，硬體 RTX 4090／24,564 MiB／CC 8.9 正常，但 CUDA/BF16 gate 拒絕，模型下載與載入均未執行 | 查核 uv／PyTorch 官方文件並鎖定 Windows cu128 index |
| 2026-07-22 | 5 | 修正 Windows CUDA wheel 路由與 processor 依賴 | Context7 與官方文件確認 `tool.uv.sources` + explicit cu128 index；PyTorch 2.10／torchvision 0.25 改鎖官方 CUDA 12.8 wheel，AutoProcessor 所需 Pillow 一併鎖定。第二次 preflight 通過後因缺 image processor dependencies 停止，補齊後不再變更模型 API | 重跑固定 acceptance |
| 2026-07-22 | 5 | 完成 RTX 4090 base + adapter acceptance | Windows／driver 591.86／PyTorch 2.10.0+cu128／CUDA 12.8／BF16 通過；固定 base revision 與 frozen step-700 adapter 成功載入。model load 84.69 秒、TTFT 1.589 秒、總生成 1.590 秒、peak allocated/reserved 8.05/11.59 GiB；strict answer `C` 符合預期，manifest 無 prompt/raw output | 歸檔內容安全證據，完成全套測試與文件提交；等待發布目標 |
| 2026-07-22 | 5 | 使用者核准精確發布目標與可見性 | GitHub `kuotunyu/tw-med-llm-qlora` 公開；Hugging Face `kuotunyu/tw-med-llm-qlora-adapter` 私人。`publication.enabled=true` 僅解鎖既有多重 gate，實際 adapter 上傳仍須 dry-run 綁定碼 | 提交設定，建立 GitHub remote 並等待 hosted CI |
| 2026-07-22 | 5 | 清理公開 snapshot 並重新關閉 HF gate | GitHub 公開 repo 已建立但尚未推送；移除完成後不再需要的離線交接工具、重複規範與 pre-4090 快照。現有 HF credential 無原 namespace 寫入權限，因此清空 adapter 目標並關閉發布 gate | 完成公開版測試；由使用者建立乾淨 Git 歷史並推送 |
| 2026-07-22 | 5 | 完成公開版清理驗收 | 移除 13 個公開雜訊檔案；四份 notebook 依安全設定重建。`pytest` 177 passed、Ruff、`uv lock --check`、四份 notebook freshness、Markdown 本機連結、credential shape 與公開 report 禁止欄位掃描全部通過 | 停在 Git 閘門；由使用者檢視變更並建立乾淨公開歷史 |

## 已知風險

- Hugging Face 的 MedQA 資料卡標示 `license: unknown`，原始 repo 標示 MIT；公開 repo 不重散布題目。
- Colab GPU 與 compute-unit 費率不保證固定，必須以當次 runtime 校準。
- TAIDE 是 gated model；權限失效時停止並要求重新接受授權。
- RTX 4090 acceptance 已以 driver 591.86、PyTorch 2.10.0+cu128 與 CUDA 12.8 通過；未來在其他 Windows 主機重跑時仍必須先通過不可跳過的 preflight，不編譯替代 wheel。
- Windows 對剛完成的大型 ZIP 或含 safetensors 目錄可能拒絕具 replace-existing 語意的改名；Phase 5 目的地原本就必須不存在，因此改用 no-overwrite `rename`，並只針對 `PermissionError` 做最多 1.25 秒的有限重試。若仍失敗會保留 `.partial`，不會誤標為完成或靜默刪除證據。
- 接手副本未包含原 `.git`，舊 parent history 只能由移轉時的 hash 證據與本執行簿追溯；目前以新 root commit 保存驗證後快照。若日後找到原機 `.git`，應另行比對，不把兩條歷史直接偽裝成同一條。
- Windows 未啟用 Developer Mode，Hugging Face cache 以無 symlink 降級模式工作，會增加磁碟用量；本次下載後仍有約 214 GiB 可用，不影響 acceptance 正確性。
- 選配 GGUF 匯出仍需在執行當下重新確認 Unsloth API 與 A100 可用磁碟；目前雙重 gate 維持關閉，尚無 GGUF 實測證據或成本數據。
- Colab 的 CUDA/PyTorch 基礎映像由平台管理；notebook 鎖定直接訓練依賴，並把實際 transitive versions 全量寫入 `pip-freeze.txt`。
- TAIDE 12B 原始 safetensors snapshot 約 27 GB；即使採 4-bit 載入仍需先保留完整下載空間，notebook 會在下載前檢查本機磁碟並保留 8 GiB 餘裕。
- Premium GPU 類型與可用性仍由 Colab 動態分配，且較高階 GPU 可能更快消耗 compute units；以 smoke manifest 的實測吞吐與帳務數據為準。
- A100 40GB 的 5.00 小時／26.49 CU 投影已分別計入純訓練、7 次 checkpoint 與 8 次 validation；仍是短跑線性外推，完整長跑可能受 runtime、Drive 或平台波動影響，因此預留 20% 至 31.78 CU。
- Drive checkpoint 採單檔 partial rename 與讀回 SHA-256 驗證，能避免把未完成檔標成 latest；但雲端掛載本身仍可能中斷，因此始終保留最近兩份驗證過的 archive，完整訓練也需預留 20% CU 緩衝。
- 完整訓練後的額外 step 703 validation 回傳 NaN；正式 adapter 因此固定使用已通過全量 validation 的 step 700。這是保守的 checkpoint 選擇，不代表最後三步必然損壞；Phase 4 必須以獨立 test 與 TMMLU+ 判定實際能力。
- Phase 4 首次 calibration 證實 vLLM 0.25.1 plain PyPI wheel 會在該 Colab 環境要求 `libcudart.so.13`。修正版改鎖官方 `0.25.1+cu129` wheel、release SHA-256 與 PyTorch `cu129` backend；本機跨平台 uv dry-run 已成功解析 216 個 Linux/Python 3.12 套件，實際原生匯入仍須由乾淨 A100 runtime 的前置 gate 驗證。若失敗會在模型與 validation 資料下載前停止，仍不接觸 test。
- 首次成功的 Phase 4 calibration 使用 box-only extractor 與 32-token 生成上限，造成 adapter 單一字母被誤判、台灣 base 在答案前截斷；該次 0% 與 2.41 小時成本投影均不得作模型或正式成本結論。parser-v3 必須以相同 validation 樣本重跑。
- parser-v3 在 256-token 上限仍有 4/20 台灣 base 輸出未形成唯一答案；正式評估會把這些視為答錯並同時報告 parse rate／token-hit，而不把 45% 解讀成純知識正確率。三模型共用相同 prompt、seed、上限與 parser。
- 正式 MedQA 中台灣 base parse rate 為 79.26%，293 次 parse failure 中有 280 次撞到 256-token 上限；TMMLU+ parse rate 為 80.75%。adapter 的增益包含 response-only 訓練帶來的格式遵循改善，不能全部當成醫療知識提升。
- 10 題代表案例揭露部分 MedQA 題目依賴缺失圖片、引用前題或含選項文字污染；總體成績衡量的是本次文字輸入協定，不等同多模態臨床推理或實際醫療安全性。

## 下一步

Phase 5 的 Windows RTX 4090 base + adapter acceptance 已完成，內容安全 manifest 與獨立驗證均通過。Windows CUDA wheel 路由已固定在 `uv.lock`，模型卡、發布 dry-run、完成度稽核器與選配 GGUF／Ollama 仍維持既有安全閘門。

GitHub `kuotunyu/tw-med-llm-qlora` 公開 repo 已建立但尚未推送；下一個必要動作是由使用者建立乾淨公開歷史、推送並確認 hosted Windows／Linux CPU CI 通過。Hugging Face 維持私人發布策略，但 `adapter_repo_id` 與發布 gate 已重新清空／關閉；確認具有寫入權限的精確 namespace 後，再執行 adapter dry-run、展示 allowlist 與一次性確認碼。GGUF export 保持關閉；HF adapter 未另行核准前不改為公開。
