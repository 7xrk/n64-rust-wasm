# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

TAS用のN64エミュレーター。エミュレーター本体は外部のエミュレーターコードを使わずにRustで書いている。

## コマンド

pnpmワークスペース。ツールチェーンは Vite+ (`vp`)。

- 開発サーバー: `pnpm dev`
  - wasmをビルドしてから `packages/front` の開発サーバーを起動する。
- wasmだけビルド: `pnpm -F n64core build:wasm`
  - Rustを変えたら必ず再ビルドする。開発サーバーはビルド済みのwasmを読む。
  - cargoは `crate/` の中で実行する。`crate/.cargo/config.toml` のSIMD指定と関数テーブルの公開は、そこで実行したときだけ効く。
- 全体ビルド: `pnpm build`
- lintとフォーマット: `npx vp check --fix packages`
  - リポジトリ直下で実行する。設定はルートの `vite.config.ts` にある。
- 型チェック: `npx tsc -p packages/front` と `npx tsc -p packages/core`
- ネイティブの起動テスト: `N64_ROM=<ROMのパス> N64_FRAMES=300 cargo test --release --manifest-path packages/core/crate/Cargo.toml boot -- --nocapture`
  - 重いので、`timeout` を付けて少ないフレーム数から始める。

テスト用ROMとムービーは `tmp/` に置く。`tmp/` はgitignore済み。

## 構成

### packages/core (npm名 `n64core`)

- `crate/`: Rust製のエミュレーター本体。外部クレートは使わない。wasm32向けにビルドし、フラットなC ABIで関数を公開する。入口は `crate/src/lib.rs`。
  - `State` は `#[repr(C)]` のPOD。セーブステートはこの構造体のバイトコピーそのもの。RDRAMは最後のフィールドで、スナップショットはその手前とRDRAMのページに分けて扱う。
  - RDRAMはホスト順のu32ワードで持つ。アドレス `a` のバイトは、ワードのビット `(3 - (a & 3)) * 8` にある。
  - CPU・RSP・RDPコマンド解析・PIF・カートリッジ保存は、すべてCPU側で決定的に動く。GPUの描画結果がCPU側のRDRAMへ戻ることはない。そのため、同じ初期状態と入力なら結果は必ず一致する。
  - RDPはピクセルを描かない。プリミティブ・状態・TMEMの記録を出力し、GPU側が描く。
  - RSPはJITで動く。`rsp_jit.rs` がIMEMの命令列をブロックごとにwasmへ変換し、`src/core.ts` がコンパイルしてコアの関数テーブルに足す。
    - 結果はインタープリターと1ビットも違わない。命令の意味を変えたら、`rsp.rs` と `rsp_jit/emit.rs` の両方を直す。
    - ネイティブのテストにはwasmのコンパイラーがないので、常にインタープリターで動く。
    - `n64_rsp_jit(0)` でJITを止められる。インタープリターと状態を比べるときに使う。
- `src/core.ts`: wasmの関数を型付きで包む `Core`。
- `src/renderer.ts` と `*.wgsl`: WebGPUのレンダラー。
  - `rdp.wgsl` はコンピュートシェーダーで、RDPの記録をフレームメモリへN64形式のまま描く。フレームメモリは、RDRAMの16ビットごとに倍率²個の画素を持つGPU上のコピーで、各画素にカバレッジとZの傾きを持つ「9ビット目」も入る。配置は `rdp.wgsl` の冒頭に書いてある。
  - 内部解像度の倍率は設定画面で1×〜4×を選び、localStorageに保存する。`fb.wgsl` が、CPUの書いたページのフレームメモリへの展開と、倍率を変えたときの移し替えをする。
  - `vi.wgsl` は2パス。1パス目がフレームメモリの解像度でVIのAA・ディザ・ディボットのフィルターをかけ、2パス目がキャンバスへ拡大してガンマをかける。
  - シェーダーは `?raw` で読み込む。

### packages/front (アプリ本体)

メインスレッドとWeb Workerに分かれる。

- メインスレッド: Preact 11 + `@preact/signals` のUI。
  - `src/ui/client.ts` がワーカーへのRPCプロキシ `emu` と、UIが描く状態のsignalを持つ。
- ワーカー: `src/emu/worker.ts` が `api` オブジェクトを `rpc.ts` で公開する。メインスレッドからの呼び出しは、すべてこのオブジェクトを通る。
  - `emulator.ts` の `Emulator` が、コア・レンダラー (OffscreenCanvas)・ムービー・スナップショット・録画を持つ。フレームループは時間予算で区切り、`exclusive()` で操作を直列化する。
- TAS関連
  - `movie.ts`: 1フレームにつき4ポート分のパッド値を持つ。ファイル形式はJSONヘッダー1行、リトルエンディアンの入力ワード、電源投入時のセーブデータ、gzipしたスナップショットの順。ムービーは埋め込んだセーブデータで電源を入れて始まる。
    - スナップショットは200フレーム以上の間隔で入る。wasmのビルドのハッシュが一致するときだけ使い、違えば入力から計算し直す。
  - Undoの単位は、ピアノロールの編集1回と、再開から一時停止までの記録1回。
  - `snapshots.ts`: 一定フレームごとに全状態を取る。変化のない4KBページは前のスナップショットと共有する。シークは最寄りのスナップショットから再エミュレートする。過去の入力を編集すると、今のフレームまで再計算する。
- 入力: キーボードとGamepad API。NSOのN64コントローラーはGamepad APIでは扱わず、WebHIDのドライバー (`nso-n64.ts`) で読む。
- `ai.ts`: Anthropic SDKをブラウザから直接使う、アプリ内のメモリサーチ用エージェント。

## ブラウザでの確認

- 内部ブラウザではなく、Claude in Chrome (chrome MCP) を使う。
- ページには `window.n64` がある。ROMやムービーのURL読み込み、ワーカーの全API、パッド値の組み立て、状態待ち、スクリーンショットを扱える。使い方は `n64.help` に出る。
