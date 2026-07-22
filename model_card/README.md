---
base_model: taide/Gemma-3-TAIDE-12b-Chat-2602
library_name: peft
pipeline_tag: text-generation
language:
  - zh
license: other
license_name: gemma-version-taide-models-license-agreement
tags:
  - gemma3
  - qlora
  - traditional-chinese
  - medical-mcq
datasets:
  - bigbio/med_qa
  - ikala/tmmluplus
---

# tw-med-llm-qlora adapter

> 本 adapter 僅供研究與教育用途，不構成醫療建議、診斷或治療依據。輸出可能錯誤；任何健康決策都應由合格醫療專業人員依完整臨床資訊判斷。

這是一個針對繁體中文醫療四選一問答進行 QLoRA 微調的 PEFT adapter。研究重點是比較同尺寸原廠 instruct、台灣在地化 base，以及相同台灣 base 加上醫療 adapter 後的能力與輸出行為。

## 基底模型與授權

- 基底：[`taide/Gemma-3-TAIDE-12b-Chat-2602`](https://huggingface.co/taide/Gemma-3-TAIDE-12b-Chat-2602)
- 固定 revision：`4de0b93b99f8b61b59c40d019fd593bdd1c42249`
- Adapter 格式：PEFT LoRA，未合併基底權重
- 基底模型是 gated repository，授權標籤為 `gemma-version-taide-models-license-agreement`；使用者必須自行接受並遵守該模型頁面的現行條款。
- 本研究 repository 的 MIT License 只涵蓋程式碼，不把 adapter、基底權重或資料重新授權為 MIT。

發布位置：[`{{HF_ADAPTER_REPO_ID}}`](https://huggingface.co/{{HF_ADAPTER_REPO_ID}})

完整重現程式：[`{{GITHUB_REPOSITORY_URL}}`]({{GITHUB_REPOSITORY_URL}})

## 資料

- 訓練／validation：`bigbio/med_qa` 的 `med_qa_tw_source`，revision `e04abdc0672c54547fa1dbe36cfefc000e4f2657`。
- 清理後固定為 train 11,248、validation 1,409；test 1,413 筆完全不進 trainer。
- 資料卡目前標示 `license: unknown`，原始 MedQA repository 標示 MIT。本 adapter repository 不重新散布題目。
- TMMLU+ revision `81d53e38340c9ade988f7fed8996da6554b504f3` 僅用於正式評估。

## 訓練設定

| 項目 | 設定 |
| --- | --- |
| 方法 | 4-bit QLoRA，response-only loss |
| LoRA | rank 16、alpha 16、dropout 0 |
| 有效 batch | 16（A100 40GB：per-device 4、gradient accumulation 4） |
| Max sequence | 2,048 |
| Optimizer | 8-bit AdamW |
| Learning rate | 5e-5，cosine，warmup ratio 0.03 |
| Epoch / seed | 1 / 3407 |
| 正式 checkpoint | step 700 |

完整訓練執行到 step 703，但額外的 step-703 validation loss 為非有限值；LoRA 權重本身通過 finite audit。為採取保守選擇，本 adapter 固定使用已通過完整 1,409 筆 validation、`eval_loss=0.227021` 的 step 700。

## 正式評估

所有結果使用 greedy decoding；無法唯一解析的答案計為錯誤並另報 parse rate。

| MedQA test（1,413 題） | Accuracy | Parse rate |
| --- | ---: | ---: |
| 原廠同尺寸 instruct | 66.17% | 99.93% |
| 台灣在地化 base | 56.40% | 79.26% |
| 台灣 base + 本 adapter | **72.05%** | **100.00%** |

Adapter 相對台灣 base 為 **+15.64 個百分點**；paired bootstrap 95% CI **[13.38, 17.98] pp**，McNemar exact test `p=6.11e-40`。

| TMMLU+ 13 科（5,573 題） | 原廠 instruct | 台灣 base | Adapter |
| --- | ---: | ---: | ---: |
| 全體 accuracy | 53.47% | 46.80% | **61.53%** |
| 8 醫學科 macro accuracy | 52.69% | 45.57% | **60.73%** |
| 5 控制科 macro accuracy | 51.73% | 44.91% | **58.41%** |

五個非醫學控制科的 adapter-base subject-macro 差異為 **+13.50 pp**，stratified paired bootstrap 95% CI **[+10.41, +16.65] pp**。下界高於預先定義的 −2.0 pp non-inferiority margin，因此只在本次指定科目與生成協定下描述為「未觀察到實質 catastrophic forgetting」。

## 使用限制

- 這是多選題研究，不是臨床安全性、長文問答、檢索、工具使用或真實照護流程評估。
- 訓練 target 只含單一答案字母；格式遵循改善是成績提升的一部分，不能把全部差距解讀為醫療知識增加。
- 部分 MedQA 題目依賴未提供圖片、引用前題或有選項文字污染；目前僅以文字協定評估。
- 正式結果只涵蓋指定資料版本、科目、seed 與 parser；不能推論到所有醫療領域或所有通用能力。
- 模型可能產生自信但錯誤、過時、偏誤或不完整的內容，也可能無法處理緊急情況。

## 推論

請先取得基底模型存取權，再依重現 repository 的 Windows RTX 4090 說明，以 4-bit NF4、BF16、SDPA 載入基底與 frozen adapter。程式會先核對 `adapter_config.json` 的基底模型，並保存不含 prompt／raw output 的本機效能 manifest。

Windows RTX 4090 acceptance 已以 PyTorch 2.10.0+cu128／CUDA 12.8 通過；固定 synthetic probe 的 model load 為 84.69 秒、first token 1.589 秒、peak allocated/reserved 8.05/11.59 GiB，嚴格答案符合預期。這是單題載入與格式驗收，不是服務吞吐 benchmark。

```powershell
uv sync --group inference
uv run tw-med-local-infer --adapter {{HF_ADAPTER_REPO_ID}} --prompt "你的 A-D 題目"
```

## 引用與重現

資料、基底模型與評估框架請依各自官方頁面的引用方式。完整設定、資料指紋、訓練曲線、統計方法、限制與 content-safe 證據位於重現 repository。
