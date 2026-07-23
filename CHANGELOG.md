# Changelog

本檔記錄公開版本的使用者可見變更。版本採用 [Semantic Versioning](https://semver.org/)。

## [0.2.0] - 2026-07-24

### Added

- 新增 public + automatic gated adapter 的最小發布、遠端逐檔驗證與匿名下載拒絕契約。
- 新增 `tw-med-verify-public-adapter` console script。
- 新增 `CITATION.cff`、package 公開連結與 README 快速入口。

### Changed

- package artifact 契約升級至五個 console scripts，並把 citation 與 changelog 納入 sdist 稽核。
- Hugging Face adapter 公開內容收斂為 adapter config、step-700 權重、模型卡與官方授權 PDF。

### Safety

- 本版不重新訓練、不修改模型權重，也不再次變更 Hugging Face 可見性。
- 模型仍為 automatic gated；匿名使用者可見 metadata，但未獲授權時不能下載檔案。

## [0.1.0] - 2026-07-23

### Added

- 首個可重現研究快照，涵蓋資料驗證、QLoRA 訓練、MedQA／TMMLU+ 雙軌評估與 Windows 本機推論。
- 發布精簡 wheel／sdist、Windows／Linux hosted CI、RTX 4090 acceptance 與內容安全研究報告。
- 完成選配 Q4_K_M／VLM GGUF 匯出及 Ollama 文字、視覺驗收；大型產物不進 Git。

[0.2.0]: https://github.com/kuotunyu/tw-med-llm-qlora/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kuotunyu/tw-med-llm-qlora/releases/tag/v0.1.0
