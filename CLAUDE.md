# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリで作業する際のガイダンスを提供します。

## ⚠️ 重要な注意事項

### 実装方針・優先順位
- **最重要**: コード量削減・効率的実装を最優先（少ないコードほど良い）
- **MUST**: 新機能実装前に既存ライブラリ・フレームワークの機能を徹底調査
- **禁止**: 場当たり的なスクラッチ実装（コードスパゲッティ化の原因）
- **調査手順**:
  1. 公式ドキュメント・API リファレンス確認
  2. 既存機能での解決可能性検討
  3. コミュニティベストプラクティス調査
  4. 既存機能で解決不可能な場合のみカスタム実装
- **実装判断基準**:
  - 3行 > 10行 > 50行（短いほど価値が高い）
  - 既存ライブラリ1行 > カスタム実装10行
  - 標準機能活用 > 独自実装
- **実装ルール**: 簡潔性・効率性・保守性・可読性を最優先
- **継承**: 複雑な実装は必ずコンテキスト化してCLAUDE.mdに記録

### パッケージ管理
- **uvとpyproject.tomlを使用してライブラリ管理を行っています**
- pipコマンドでの直接インストールは避けてください
- 新しいライブラリの追加: `uv add <package>`
- 開発用ライブラリの追加: `uv add --dev <package>`
- ライブラリの削除: `uv remove <package>`

### ソース管理・ファイル配置ルール【厳格遵守】
- **本格実装**: src/配下に置くこと（.pyファイル）
- **プロトタイプ・実験**: notebooks/配下に置くこと（.ipynbファイル）
- **テストファイル**: tests/配下に置くこと（test_*.pyファイル）
- **一時実行ファイル**: scripts/配下に置くこと（実験・デバッグ用）

#### 🚫 絶対禁止事項
- **test_*.py ファイルをnotebooks/やsrc/配下に作成することは厳禁**
- **テストファイルは必ずtests/配下に配置**
- **ルート直置きスクリプトの作成は禁止**
- **一時的なスクリプト・ノートブックは必ずscripts/配下に配置**
- **上記違反は即座に修正すること**

### VSCode設定・Import規則
- **VSCodeでルートパス基準のimport設定を使用**
- 相対パスやsys.path.append()は使用せず、VSCodeのワークスペース設定に依存

### run.pyとnotebook同期ルール
- **最重要**: `src/mlops/run.py`と`notebooks/mlops_experiment_v2.ipynb`のコア機能は常に1:1で同期
- **禁止**: 片方にのみ機能を実装することは厳禁
- **必須**: 機能追加・修正時は必ず両方を同期更新
- **検証**: 実行結果が同じになることを確認
- **対象**: MLOpsパイプライン、サンプリング、最適化、評価、可視化の全コア機能

### Git開発フロー（Git Flow拡張版）
- **ブランチ戦略**:
  - `main`: 本番環境用の安定版ブランチ
  - `develop`: 開発統合ブランチ（日常的な開発はここから分岐）
  - `feature/機能名`: 機能開発用ブランチ
  - `bugfix/バグ名`: 開発中のバグ修正用ブランチ（develop起点）
  - `hotfix/緊急修正名`: 本番の緊急バグ修正用ブランチ（main起点）

- **ブランチ命名規則**:
  - `feature/機能概要`: 新機能・機能拡張 (例: feature/sampling, feature/cv-strategy)
  - `bugfix/バグ概要`: 開発版のバグ修正 (例: bugfix/readme-corruption, bugfix/import-error)
  - `hotfix/緊急修正概要`: 本番緊急修正 (例: hotfix/critical-crash, hotfix/security-fix)

- **開発手順**:

  **1. 機能開発 (feature)**:
  ```bash
  git checkout develop
  git pull origin develop
  git checkout -b feature/新機能名
  # 開発後
  git checkout develop
  git merge feature/新機能名
  git push origin develop
  git branch -d feature/新機能名
  ```

  **2. バグ修正 (bugfix)**:
  ```bash
  git checkout develop
  git pull origin develop
  git checkout -b bugfix/バグ名
  # 修正後
  git checkout develop
  git merge bugfix/バグ名
  git push origin develop
  git branch -d bugfix/バグ名
  ```

  **3. 緊急修正 (hotfix)**:
  ```bash
  git checkout main
  git pull origin main
  git checkout -b hotfix/緊急修正名
  # 修正後
  git checkout main
  git merge hotfix/緊急修正名
  git push origin main
  # developにも反映
  git checkout develop
  git merge hotfix/緊急修正名
  git push origin develop
  git branch -d hotfix/緊急修正名
  ```

- **コミットメッセージ規則**:
  - `feat:` 新機能追加
  - `fix:` バグ修正
  - `hotfix:` 緊急修正
  - `docs:` ドキュメント更新
  - `refactor:` リファクタリング
  - `test:` テスト追加・修正


### 開発手法・プロセス

#### 開発サイクル（3段階プロセス）
1. **プロトタイプフェーズ** - 概要fix・試行錯誤
   - 📓 **notebook中心**でクイック実装・動作確認
   - 📄 main.pyまたはnotebook（実行可能ならnotebook優先）
   - 🎯 機能の概要固定・アイデア検証
   - ⚡ 高速試行錯誤、テスト不要
   - ✅ **前提条件**: notebookが通しで実行できること

2. **コンポーネント分離フェーズ** - 構造化・品質確保
   - 🔧 src/配下への.pyファイル分離
   - 🧪 **TDD導入開始**（テスト → 実装 → リファクタリング）
   - 📋 インターフェース設計・モジュール分離
   - 🏗️ アーキテクチャ構築

3. **拡張フェーズ** - 機能拡張・本格運用
   - 🚀 機能追加・性能最適化
   - 📊 継続的なテスト・カバレッジ向上
   - 📚 ドキュメント整備

#### プロトタイプフェーズ詳細ルール
- 📓 **notebooks/配下に.ipynbファイル作成**
- 🔄 **必須**: notebook全体が通しで実行できること
- 💡 **目的**: アイデア検証・機能概要の固定
- 🚫 **制限**: コード品質・テストは不問
- ✅ **完了条件**: 想定機能が動作することを確認

#### コンポーネント分離ルール
- 🔧 **移行方法**: notebookのセル単位でsrc/配下の.pyファイルに分離
- 📋 **インターフェース設計**: 関数・クラスの責務明確化
- 🧪 **TDD開始**: 分離したコンポーネントから順次テスト作成
- 📁 **配置**: src/配下の適切なモジュール構造
- 🔗 **統合**: notebookからsrc/配下の関数をimportして動作確認

#### フェーズ移行判断基準
1. **プロトタイプ → コンポーネント分離**
   - ✅ 機能の基本動作確認完了
   - ✅ notebook全体が通しで実行可能
   - ✅ 次の開発ステップが明確

2. **コンポーネント分離 → 拡張**
   - ✅ 主要コンポーネントのsrc/分離完了
   - ✅ 基本テストケース作成完了
   - ✅ アーキテクチャ構造確定

#### テスト駆動開発（TDD）
- **コンポーネント分離フェーズから開始**
- テスト → 実装 → リファクタリングの順で進める
- `uv run pytest` でテスト実行
- テストカバレッジは `uv run pytest --cov=src tests/` で確認

#### 3層構造による要件管理
1. **docs/specs/def.md（要件定義）** - 最重要レイヤー
   - システム全体の設計思想と要件
   - 変更時は必ず確認が必要
   - アーキテクチャ・技術選択の根拠

2. **仕様書（`docs/specs/`）** - 実装詳細レイヤー
   - 各モジュールの詳細仕様
   - API設計、クラス構造、インターフェース
   - docs/specs/def.mdから派生した具体的な実装方針

3. **コード実装** - 実装レイヤー
   - 仕様書に基づく実装
   - テストコードも含む
   - docs/specs/def.md・仕様書との整合性を保つ

#### 変更管理原則
- **docs/specs/def.mdの変更 = 設計変更** → 慎重な検討が必要
- 仕様書の変更 → docs/specs/def.mdとの整合性確認
- コード変更 → 仕様書・テストとの整合性確認
- 全ての変更は対応するテストの更新も含む

#### テストコード管理
- **全てのテストは `tests/` ディレクトリ以下に配置**
- テストファイルの命名: `test_<機能名>.py`
- ディレクトリ構造例:
  ```
  tests/
  ├── unit/                    # 単体テスト（個別機能）
  ├── integration/             # 統合テスト（パイプライン全体）
  └── utils/                   # ユーティリティテスト
  ```
- **ルートディレクトリにテストファイルを作成しない**
- 実行方法: `uv run pytest tests/` または機能別実行

## 開発コマンド

### パッケージインストール
```bash
# 基本的な依存関係をインストール
pip install -e .

# オプションのML ライブラリと一緒にインストール
pip install -e ".[ml]"

# 全てのオプション依存関係をインストール（開発時推奨）
pip install -e ".[all]"

# 開発ツールをインストール
pip install -e ".[dev]"
```

### コード品質管理
```bash
# コードフォーマット
black src/
isort src/

# 型チェック
mypy src/

# リンティング
flake8 src/

# テスト実行
pytest
pytest --cov=src tests/
```
