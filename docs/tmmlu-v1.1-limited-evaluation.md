# TMMLU+ v1.1 有限評估更新

本文件記錄在既有 v0.2.0 成果之上新增的一份**版本化評估**：固定原本三組模型，檢查 TMMLU+ 上游
v1.1 修訂後，醫學科目的改善與五個非醫學控制科目的 non-inferiority 結論是否仍成立，以及分數差異
有多少來自答案修訂、題目集合變動與格式解析失敗。本次不重訓、不新增 checkpoint、不改判準，
也不用新版 test 挑選模型、prompt 或 checkpoint；MedQA 未因此更新，不重跑。

> 免責聲明：研究與教育用途，所有結果不構成臨床診斷、醫療建議或治療依據。

## 1. 固定不變的比較對象

| 項目 | 身份 | 來源核對 |
|---|---|---|
| 原始 Instruct baseline | `google/gemma-3-12b-it` @ `96b6f1eccf38110c56df3a15bffe176da04bfd80` | `configs/project.toml` `[models.primary].baseline_*`；Phase 4 manifest |
| TAIDE base | `taide/Gemma-3-TAIDE-12b-Chat-2602` @ `4de0b93b99f8b61b59c40d019fd593bdd1c42249` | `[models.primary]`；Phase 4 manifest |
| Step 700 adapter | `steven0226/tw-med-llm-qlora-adapter` @ `b1d8f74291da75d0719b5a3ea0d088ee8236e096` | `adapter_model.safetensors` sha256 `10c88d77…a666` 與本機 Phase 5 轉移收據一致；`adapter_config.json` sha256 `bc062f9d…3999` 與 Phase 4 manifest 一致 |

Prompt、parser、生成設定均沿用 Phase 4 正式契約（zero-shot、固定 system prompt、題幹＋依序選項、
strict A–D 或單一 `\boxed{}` parser、temperature 0、max_tokens 256、選項洗牌 seed 3407、
256-token 截斷計為錯）。

## 2. 資料版本

| 版本 | Revision | 說明 |
|---|---|---|
| v1.0（原研究） | `81d53e38340c9ade988f7fed8996da6554b504f3` | `[data.tmmluplus]`，凍結證據所用 |
| v1.1（本次） | `94d86f1d013d59f389b710a902af30b7a0aa8c8f` | `configs/tmmlu_v1_1.toml`；上游 tag `v1.1`；執行時以 Hub refs 驗證 tag 指向此 commit，不使用 mutable `main` |

只讀取正式 config 映射的 13 科 `*_test.csv`（醫學 8 科 + 控制 5 科），實際列數保存在
`run-manifest.json` 的 `dataset_files`。上游說明表數字與實際 CSV 計數的落差不影響本研究，因為
所有分母都來自實際讀入的列。

## 3. 新舊題目對照方法

- 配對鍵：（科目、題幹、依序 A/B/C/D 選項文字）**精確**比對，multiset 計數。
- 分類：`unchanged`（輸入與答案未變）、`answer_revised`（輸入未變、答案修訂）、
  `removed_or_modified`（v1.0 列在 v1.1 無同內容列）、`added_or_modified`（v1.1 列在 v1.0 無同內容列）、
  `duplicate_ambiguous`（任一版本同內容出現多次）。
- 不以列號、seed 或模糊比對判定同題；空白差異與選項重排只作診斷欄位，不用於計分。
- 本專案 ID 含 revision 與 source-row，新舊 ID 不直接 join；mapping 檔不含題目文字。

## 4. 證據種類

1. **歷史輸出依新版答案／題組重新計分**（`historical_outputs_rescored_under_v1_1`）：
   Phase 4 為 zero-shot，內容完全相同的題目即代表模型看到相同輸入；公開逐題檔保存了每題
   prediction（洗牌後字母空間），因此以同一洗牌映射把 v1.1 答案轉入該空間後重新計分。
   這不是新執行的完整 v1.1 評估。
2. **本機新推論探針**（`new_inference_separate_backend_probe`）：Windows RTX 4090、
   Transformers + bitsandbytes NF4 + SDPA，與 Phase 4 的 vLLM/A100 為不同 backend，故另列為
   新實驗，僅用來（a）補做 v1.1 未覆蓋題、（b）量測與歷史 prediction 的一致率，不併入主表。

## 5. 結果（自動產生）

<!-- generated:tmmlu-v1.1:begin -->
_下列表格由 `reports/tmmlu-v1.1/20260913T170025Z/analysis.json` 逐題證據自動產生；證據種類：`historical_outputs_rescored_under_v1_1`；凍結數字重現檢查：`passed`。_

#### 結論摘要（自動產生）

- **覆蓋率**：v1.1 的 5500 題中有 5495 題與 v1.0 內容完全相同，可由歷史輸出重新計分；1 題為 v1.1 新增或修改（未覆蓋），4 列因重複內容而列為歧義；v1.0 有 74 題在 v1.1 無同內容列，28 題答案修訂。
- **醫學增益仍成立**：保留題組 × v1.1 答案下，Adapter 醫學 8 科 macro 61.30%，對 TAIDE Base +15.38 pp [+13.71, +17.07]（v1.0 全量：+15.16 pp [+13.46, +16.83]），對原始 Instruct +8.35 pp [+6.35, +10.32]（v1.0 全量：+8.04 pp [+5.99, +9.98]）。
- **控制科目 non-inferiority 仍成立**：控制 5 科 Adapter − Base subject-macro +13.65 pp [+10.47, +16.85]，95% CI 下界高於預先定義的 -2.0 pp 門檻，判定 `no_material_forgetting`（v1.0 全量：+13.50 pp [+10.41, +16.65]，`no_material_forgetting`）。
- **差異來源**：Adapter 13 科整體由 61.53% 變為 62.04%（+0.51 pp），其中刪題效應 +0.35 pp、答案修訂效應 +0.16 pp；28 題答案修訂中 Adapter 有 16 題轉對、7 題轉錯。
- **格式解析因素未消除**：TAIDE Base 在保留題組仍有 1013 題 256-token 截斷與 50 題其他解析失敗（parse rate 80.66%），皆計為錯；Adapter 與 Base 的差距同時包含格式服從與知識，表 7 的皆可解析子集只是描述性切片。
- **同一輸出、僅答案版本不同**：三模型在保留題組上 v1.1 − v1.0 的 13 科整體差值分別為 原始 Instruct +0.11 pp [-0.05, +0.27]、TAIDE Base +0.09 pp [-0.04, +0.24]、Step 700 Adapter +0.16 pp [+0.00, +0.35]。

#### 表 1：新舊題目覆蓋率

| 範圍 | v1.0 題數 | v1.1 題數 | 內容完全相同 | 其中答案修訂 | v1.0 刪除或修改 | v1.1 新增或修改（未覆蓋） | 重複內容歧義 | 可重新計分的 v1.1 題數 |
|---|---|---|---|---|---|---|---|---|
| 13 科全部 | 5573 | 5500 | 5495 | 28 | 74 | 1 | 4 | 5495 |
| 醫學 8 科 | 4187 | 4151 | 4150 | 13 | 37 | 1 | 0 | 4150 |
| 控制 5 科 | 1386 | 1349 | 1345 | 15 | 37 | 0 | 4 | 1345 |

#### 表 2：三種計分方式下的整體指標

| 計分方式 | 模型 | 題數 | 13 科整體 | 醫學 8 科 macro | 控制 5 科 macro | Parse rate | 256-token 截斷 | 其他解析失敗 |
|---|---|---|---|---|---|---|---|---|
| v1.0 全量（凍結結果重算） | 原始 Instruct | 5573 | 53.47% | 52.69% | 51.73% | 99.89% | 0 | 6 |
| v1.0 全量（凍結結果重算） | TAIDE Base | 5573 | 46.80% | 45.57% | 44.91% | 80.75% | 1023 | 50 |
| v1.0 全量（凍結結果重算） | Step 700 Adapter | 5573 | 61.53% | 60.73% | 58.41% | 100.00% | 0 | 0 |
| 保留題組 × v1.0 答案 | 原始 Instruct | 5495 | 53.76% | 52.82% | 52.30% | 99.89% | 0 | 6 |
| 保留題組 × v1.0 答案 | TAIDE Base | 5495 | 47.04% | 45.72% | 45.46% | 80.66% | 1013 | 50 |
| 保留題組 × v1.0 答案 | Step 700 Adapter | 5495 | 61.87% | 61.02% | 58.83% | 100.00% | 0 | 0 |
| 保留題組 × v1.1 答案（主結果） | 原始 Instruct | 5495 | 53.87% | 52.95% | 52.48% | 99.89% | 0 | 6 |
| 保留題組 × v1.1 答案（主結果） | TAIDE Base | 5495 | 47.13% | 45.92% | 45.59% | 80.66% | 1013 | 50 |
| 保留題組 × v1.1 答案（主結果） | Step 700 Adapter | 5495 | 62.04% | 61.30% | 59.23% | 100.00% | 0 | 0 |
| 敏感度：加入依列序配對的重複列 | 原始 Instruct | 5499 | 53.86% | 52.95% | 52.50% | 99.89% | 0 | 6 |
| 敏感度：加入依列序配對的重複列 | TAIDE Base | 5499 | 47.12% | 45.92% | 45.42% | 80.65% | 1014 | 50 |
| 敏感度：加入依列序配對的重複列 | Step 700 Adapter | 5499 | 62.05% | 61.30% | 59.29% | 100.00% | 0 | 0 |

#### 表 3：各科結果（保留題組 × v1.1 答案）

| 群組 | 科目 | 題數 v1.0 → v1.1 覆蓋 | Instruct | Base | Adapter | Adapter Δ vs v1.0 | Base parse rate | Base 截斷 / 其他失敗 |
|---|---|---|---|---|---|---|---|---|
| 醫學 | basic_medical_science | 954 → 944 | 62.39% | 54.24% | 74.79% | +0.36 pp | 75.21% | 211 / 23 |
| 醫學 | clinical_psychology | 125 → 125 | 65.60% | 60.00% | 72.00% | +0.80 pp | 88.80% | 14 / 0 |
| 醫學 | dentistry | 399 → 397 | 51.39% | 47.10% | 61.71% | +0.31 pp | 77.08% | 89 / 2 |
| 醫學 | occupational_therapy_for_psychological_disorders | 543 → 542 | 65.87% | 53.51% | 72.51% | +0.50 pp | 80.81% | 95 / 9 |
| 醫學 | optometry | 920 → 915 | 39.34% | 35.08% | 45.03% | +0.24 pp | 79.02% | 191 / 1 |
| 醫學 | pharmacology | 577 → 571 | 59.89% | 43.78% | 63.40% | +0.31 pp | 74.78% | 140 / 4 |
| 醫學 | pharmacy | 391 → 379 | 39.05% | 34.30% | 43.54% | +1.08 pp | 81.00% | 68 / 4 |
| 醫學 | traditional_chinese_medicine_clinical_medicine | 278 → 277 | 40.07% | 39.35% | 57.40% | +0.93 pp | 75.45% | 67 / 1 |
| 控制 | chinese_language_and_literature | 199 → 195 | 43.59% | 36.92% | 50.26% | +0.01 pp | 91.28% | 17 / 0 |
| 控制 | geography_of_taiwan | 768 → 754 | 61.67% | 62.47% | 72.55% | +0.15 pp | 90.58% | 67 / 4 |
| 控制 | logic_reasoning | 139 → 130 | 35.38% | 28.46% | 30.00% | +2.66 pp | 93.85% | 7 / 1 |
| 控制 | computer_science | 174 → 168 | 73.81% | 54.17% | 75.00% | +0.86 pp | 82.14% | 29 / 1 |
| 控制 | general_principles_of_law | 106 → 98 | 47.96% | 45.92% | 68.37% | +0.44 pp | 81.63% | 18 / 0 |

#### 表 4：配對統計與預先定義判準

| 計分方式 | 比較 | 配對題數 | 觀察差值 | Bootstrap 95% CI | 判讀 |
|---|---|---|---|---|---|
| v1.0 全量（凍結結果重算） | 控制 5 科 Adapter − Base（subject-macro） | 1386 | +13.50 pp | [+10.41, +16.65] | −2 pp 判準：no_material_forgetting |
| v1.0 全量（凍結結果重算） | 醫學 8 科 Adapter − Base（subject-macro） | 4187 | +15.16 pp | [+13.46, +16.83] | McNemar p = 1.63e-99（843 進步 / 188 退步） |
| v1.0 全量（凍結結果重算） | 醫學 8 科 Adapter − Instruct（subject-macro） | 4187 | +8.04 pp | [+5.99, +9.98] | McNemar p = 1.28e-21（774 進步 / 442 退步） |
| v1.0 全量（凍結結果重算） | 13 科整體 Adapter − Base（micro） | 5573 | +14.73 pp | [+13.53, +15.93] | McNemar p = 8.41e-124（1059 進步 / 238 退步） |
| v1.0 全量（凍結結果重算） | 13 科整體 Adapter − Instruct（micro） | 5573 | +8.06 pp | [+6.68, +9.44] | McNemar p = 1.42e-29（1022 進步 / 573 退步） |
| 保留題組 × v1.1 答案（主結果） | 控制 5 科 Adapter − Base（subject-macro） | 1345 | +13.65 pp | [+10.47, +16.85] | −2 pp 判準：no_material_forgetting |
| 保留題組 × v1.1 答案（主結果） | 醫學 8 科 Adapter − Base（subject-macro） | 4150 | +15.38 pp | [+13.71, +17.07] | McNemar p = 1.31e-100（844 進步 / 186 退步） |
| 保留題組 × v1.1 答案（主結果） | 醫學 8 科 Adapter − Instruct（subject-macro） | 4150 | +8.35 pp | [+6.35, +10.32] | McNemar p = 1.10e-22（772 進步 / 433 退步） |
| 保留題組 × v1.1 答案（主結果） | 13 科整體 Adapter − Base（micro） | 5495 | +14.90 pp | [+13.70, +16.14] | McNemar p = 2.87e-124（1053 進步 / 234 退步） |
| 保留題組 × v1.1 答案（主結果） | 13 科整體 Adapter − Instruct（micro） | 5495 | +8.17 pp | [+6.79, +9.57] | McNemar p = 4.76e-30（1009 進步 / 560 退步） |

#### 表 5：差異分解（刪題 vs 答案修訂）

| 模型 | 指標 | v1.0 全量 | 保留題組 × v1.0 答案 | 保留題組 × v1.1 答案 | 刪題效應 | 答案修訂效應 | 合計變化 |
|---|---|---|---|---|---|---|---|
| 原始 Instruct | 13 科整體 | 53.47% | 53.76% | 53.87% | +0.29 pp | +0.11 pp | +0.40 pp |
| 原始 Instruct | 醫學 8 科 macro | 52.69% | 52.82% | 52.95% | +0.14 pp | +0.13 pp | +0.26 pp |
| 原始 Instruct | 控制 5 科 macro | 51.73% | 52.30% | 52.48% | +0.57 pp | +0.18 pp | +0.75 pp |
| TAIDE Base | 13 科整體 | 46.80% | 47.04% | 47.13% | +0.25 pp | +0.09 pp | +0.34 pp |
| TAIDE Base | 醫學 8 科 macro | 45.57% | 45.72% | 45.92% | +0.16 pp | +0.20 pp | +0.35 pp |
| TAIDE Base | 控制 5 科 macro | 44.91% | 45.46% | 45.59% | +0.55 pp | +0.13 pp | +0.67 pp |
| Step 700 Adapter | 13 科整體 | 61.53% | 61.87% | 62.04% | +0.35 pp | +0.16 pp | +0.51 pp |
| Step 700 Adapter | 醫學 8 科 macro | 60.73% | 61.02% | 61.30% | +0.29 pp | +0.28 pp | +0.57 pp |
| Step 700 Adapter | 控制 5 科 macro | 58.41% | 58.83% | 59.23% | +0.42 pp | +0.41 pp | +0.82 pp |

#### 表 6：同一模型、同一輸出、僅答案版本不同

| 模型 | 指標 | 配對題數 | 答案修訂題數（結果翻轉） | v1.1 − v1.0 | Bootstrap 95% CI |
|---|---|---|---|---|---|
| 原始 Instruct | 13 科整體 | 5495 | 28（轉對 14 / 轉錯 8） | +0.11 pp | [-0.05, +0.27] |
| 原始 Instruct | 醫學 8 科 macro | 5495 | 28（轉對 14 / 轉錯 8） | +0.13 pp | [-0.08, +0.40] |
| 原始 Instruct | 控制 5 科 macro | 5495 | 28（轉對 14 / 轉錯 8） | +0.18 pp | [-0.74, +1.10] |
| TAIDE Base | 13 科整體 | 5495 | 28（轉對 10 / 轉錯 5） | +0.09 pp | [-0.04, +0.24] |
| TAIDE Base | 醫學 8 科 macro | 5495 | 28（轉對 10 / 轉錯 5） | +0.20 pp | [+0.01, +0.47] |
| TAIDE Base | 控制 5 科 macro | 5495 | 28（轉對 10 / 轉錯 5） | +0.13 pp | [-0.51, +0.82] |
| Step 700 Adapter | 13 科整體 | 5495 | 28（轉對 16 / 轉錯 7） | +0.16 pp | [+0.00, +0.35] |
| Step 700 Adapter | 醫學 8 科 macro | 5495 | 28（轉對 16 / 轉錯 7） | +0.28 pp | [+0.06, +0.56] |
| Step 700 Adapter | 控制 5 科 macro | 5495 | 28（轉對 16 / 轉錯 7） | +0.41 pp | [-0.49, +1.33] |

#### 表 7：三模型皆可解析子集（描述性，未消除格式混淆）

| 計分方式 | 模型 | 三模型皆解析題數 | 13 科整體 | 醫學 8 科 macro | 控制 5 科 macro |
|---|---|---|---|---|---|
| v1.0 全量（凍結結果重算） | 原始 Instruct | 4494 | 53.23% | 52.20% | 50.91% |
| v1.0 全量（凍結結果重算） | TAIDE Base | 4494 | 57.97% | 57.58% | 51.32% |
| v1.0 全量（凍結結果重算） | Step 700 Adapter | 4494 | 60.93% | 59.77% | 57.60% |
| 保留題組 × v1.1 答案（主結果） | 原始 Instruct | 4426 | 53.73% | 52.54% | 51.84% |
| 保留題組 × v1.1 答案（主結果） | TAIDE Base | 4426 | 58.45% | 58.06% | 52.18% |
| 保留題組 × v1.1 答案（主結果） | Step 700 Adapter | 4426 | 61.39% | 60.31% | 58.37% |

#### 表 8：本機新推論探針（獨立實驗，backend = `windows-rtx4090-transformers-bnb-nf4`）

_證據種類：`new_inference_separate_backend_probe`；此表不得與表 2–7 的歷史重計分合併解讀。_

| 模型 | 探針題數（歷史 prompt） | 與歷史 prediction 一致 | Wilson 95% CI | 同 backend 上限（full vs stability-3407） | prompt_tokens 相同 | 探針 v1.1 正確率 | v1.1 未覆蓋題 | 其中答對 |
|---|---|---|---|---|---|---|---|---|
| 原始 Instruct | 195 | 192 / 195 = 98.46% | [95.6%, 99.5%] | n/a | 195 | 54.87% | 1 | 0 |
| TAIDE Base | 195 | 190 / 195 = 97.44% | [94.1%, 98.9%] | 1278 / 1300 = 98.31% | 195 | 49.74% | 1 | 0 |
| Step 700 Adapter | 195 | 189 / 195 = 96.92% | [93.5%, 98.6%] | 1271 / 1300 = 97.77% | 195 | 61.54% | 1 | 0 |
<!-- generated:tmmlu-v1.1:end -->

## 6. 結論

第 5 節開頭的「結論摘要」由 `analysis.json` 自動產生，數字以該處為準。就本次核心問題：

1. **醫學科目的改善仍成立。** 在保留題組 × v1.1 答案下，Step 700 adapter 對 TAIDE base 與對原始
   Instruct 的醫學 8 科 subject-macro 差值及其 bootstrap 95% CI 全數為正，且與 v1.0 全量結果相差不到
   0.4 pp；McNemar 檢定的方向與顯著性不變（表 4）。
2. **五個控制科目仍符合原先的 −2 pp non-inferiority 判準。** adapter − base 的 subject-macro 95% CI 下界
   遠高於 −2 pp，判定維持 `no_material_forgetting`（表 4）。判準、bootstrap 次數與 seed 皆沿用原研究。
3. **分數差異來源。** 三模型 13 科整體在 v1.1 重計分下的變化都在 1 pp 以內，且刪題效應與答案修訂效應
   大小相當（表 5）；同一輸出僅換答案版本的差值 95% CI 幾乎都跨越 0（表 6）。答案修訂集中在
   logic_reasoning（130 題中 11 題），是唯一單科 adapter 變動超過 2 pp 的科目（表 3）。格式解析失敗
   （TAIDE base 約 19% 的 256-token 截斷）在新舊版本間幾乎不變，因此不是新舊差異的來源，但仍是
   adapter − base 差距的組成部分之一（表 2、表 7）。

以上結論屬「歷史輸出依 v1.1 答案／題組重新計分」，覆蓋 v1.1 5,500 題中的 5,495 題；未覆蓋的 1 題與
本機探針結果（表 8，若存在）屬不同 backend 的新實驗，不改變上述判定。

## 7. 限制

- 保留題組的重計分沿用歷史輸出，能回答「同一輸出在新答案下的分數」，不能替代同 backend 的
  完整 v1.1 重新執行；v1.1 新增／修改而未覆蓋的題目只在探針實驗中以不同 backend 補做。
- TAIDE base 的 parse rate 約 80%，其與 adapter 的差距同時包含格式服從與知識；只看三模型皆可
  解析的子集屬描述性切片，不能宣稱已消除格式因素的因果混淆。
- 探針 backend 與正式評估不同；同 backend、同 prompt 在歷史資料內的 prediction 一致率
（`tmmlu-full` vs `tmmlu-stability-3407`）已作為可比上限列出。
- 重複內容列（general_principles_of_law）在主分析中排除，僅提供依列序配對的敏感度結果。

## 8. 未完成事項與 Colab 方案

- **判定：不需要付費重跑。** 重計分已覆蓋 v1.1 5,500 題中的 5,495 題，且本機探針對歷史 prompt 的
  prediction 一致率落在同 backend 的非決定性上限附近，核心問題已能回答；同 backend 全量重跑只會
  多花費用而不改變判定，除非日後需要正式的完整 v1.1 數字才考慮。
- **同 backend 全量 v1.1 評估**未執行（Windows 無 vLLM）。若日後需要：以 Phase 4 相同 Colab A100 +
  vLLM 0.25.1 + bitsandbytes 契約，只跑 `tmmlu-full` 3 模型 × 5,500 題 = 16,500 次生成。
  以 Phase 4 實測（28,758 次 / 2.542 h / 13.47 CU）等比估算約 1.5 h、約 8 CU；加 20% 緩衝約 10 CU。
  僅補做 1 題未覆蓋題亦需啟動 A100 與載入兩個 12B 模型，約 0.5 h、約 2.7 CU。需另建 v1.1 專用
  notebook builder（不得覆寫 `evaluate_phase4_full.ipynb`），並先取得使用者確認付費用量。

## 9. 證據路徑

- `reports/tmmlu-v1.1/<run-id>/mapping.json`：新舊題目對照（無題目文字）。
- `reports/tmmlu-v1.1/<run-id>/analysis.json`：重新計分與統計。
- `reports/tmmlu-v1.1/<run-id>/report.md`：由 analysis 生成的表格。
- `reports/tmmlu-v1.1/<run-id>/run-manifest.json`：資料版本、檔案 sha256、prompt/parser 契約、程式版本。
- `reports/tmmlu-v1.1/<run-id>/local-inference/`：探針公開逐題列（含 prompt/raw output 的 sha256）、
  runtime 與 probe-analysis。
- `reports/private/tmmlu-v1.1/<run-id>/`（gitignored）：探針私有逐題列（題目、prompt、raw output）。
- 歷史證據：`reports/phase4/full/20260722T070936Z-phase4-full-public.zip`（sha256 `1804c26d…4f76`）未變動。
