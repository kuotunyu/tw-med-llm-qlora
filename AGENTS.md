# 環境與工作守則 v4
- 環境：Windows 11 原生（非 WSL），GPU RTX 4090 (24GB)，Python 3.11，一律用 uv 管理虛擬環境；終端機為 PowerShell（或 Git Bash），檔案路徑一律以 pathlib 處理、不寫死斜線
- API 金鑰一律讀 .env（GOOGLE_API_KEY, OPENAI_API_KEY, HF_TOKEN, DISCORD_WEBHOOK_URL），.env 進 .gitignore；必須同步維護 .env.example（不含真值）

## Windows 相容性守則
- 地端 LLM 服務一律用 Ollama（vLLM 不支援 Windows，不得採用）
- 向量庫一律 chromadb（faiss 在 Windows 相容性差，不得採用）
- 需要 flash-attn 的模型（如 ColQwen2）改用 sdpa attention 實作
- 任何套件若無 Windows wheel 或安裝失敗，先回報並提出替代方案讓我選，不要硬編譯

## 模型政策
- 雲端模型字串一律由 .env 設定（GEMINI_MODEL 預設 gemini-3.1-flash-lite；GEMINI_LITE_MODEL 預設 gemini-2.5-flash-lite；OPENAI_MODEL 預設 gpt-5-mini），程式碼不得寫死；使用前以官方文件確認字串現行有效，失效則回報並建議替代
- 分工：主力生成/Agent/VLM 用 GEMINI_MODEL；大量一次性前處理（chunk 摘要、合成資料）用 GEMINI_LITE_MODEL；評審與第二供應商用 OPENAI_MODEL
- 地端 LLM 首選 taide/Gemma-3-TAIDE-12b-Chat-2602（Ollama 模型名 taide-gemma3-12b 已建好）；輕量情境用 twinkle-ai/gemma-3-4B-T1-it 或 twinkle-ai/Llama-3.2-3B-F1-Instruct
- Embedding 首選 taide/embeddinggemma-GTAIDE-300m-2605（sentence-transformers 載入；務必依 EmbeddingGemma 規範，query 與 document 使用各自對應的 prompt_name 編碼，否則檢索品質嚴重下降）
- Reranker 用 BAAI/bge-reranker-v2-m3（台灣生態系目前無本土 reranker，README 可註記此觀察）
- 所有用到模型的模組必須「可切換」，並保留與基準模型（BAAI/bge-m3、google/gemma-3-12b-it）同一評估集的對照能力

## LangChain 與 Context7 規範
- Agent、RAG、工具調用與應用層 LLM 編排一律用 LangChain 1.x 穩定版現行 API：create_agent、middleware、init_chat_model、init_embeddings、LCEL；禁止使用已移入 langchain-classic 的舊式 chains/AgentExecutor
- 模型訓練、tokenization、Unsloth、Transformers、PEFT、TRL 與評估 runner 使用各專案官方原生 API，不為了形式統一而套用 LangChain
- 撰寫或修改任何使用外部套件（LangChain、LangGraph、sentence-transformers、ultralytics、unsloth 等）的程式前，自動使用 Context7 MCP（resolve-library-id + query-docs）確認當前版本 API，不需要我明講；Context7 查不到就讀官方文件
- 評估框架若遇相容性問題（如 RAGAS 與 LangChain 1.x 衝突），改用 deepeval 或自製 LLM-as-judge 腳本，並在 README 說明
- requirements 以 uv lock 鎖定，README 記錄關鍵套件版本

## 公開文案守則（重要）
- README、commit 訊息、模型卡、HF Space 說明中，禁止出現任何特定公司名稱或其產品名；動機一律以個人/家庭經驗、高齡社會議題、台灣開源生態興趣來撰寫
- README 開頭要有一段第一人稱動機（我會提供或請你依專案主題擬一段自然的個人動機供我修改）

## Git 與交付規範
- Conventional Commits 風格；每完成一個小功能就 commit，禁止巨型單一 commit
- commit 訊息預設使用正體中文；Conventional Commit 的 type/scope 與技術專有名詞保留原文
- 模型權重與大型資料不進 git：權重上傳 Hugging Face，資料寫下載腳本
- repo 附 MIT LICENSE；使用政府開放資料處在 README 註明資料來源與授權
- README.md 繁體中文，必含：個人動機段、mermaid 架構圖、模型選型說明（台灣模型 vs 基準模型對照表）、快速開始、評估結果、成本估算
- 分 Phase 開發：每個 Phase 先跑通、給我看結果、經我確認再進下一步
- 外部資源先驗證可用性；gated model 下載失敗時提示我去 HF 接受授權，不要換模型硬做
- 會花錢的批次 API 呼叫，先估算成本印給我確認
- 寫 pytest 基本測試並執行通過
