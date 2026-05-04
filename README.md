# BidKing Analyzer RS

竞拍之王估价器的 Rust 开源实现。项目包含核心估价模型、egui 桌面 GUI、Windows 全局快捷键、屏幕区域截图，以及基于 PP-OCRv4 ONNX 的视觉扫描。

本项目是独立实现，目标是提供一个可维护、可构建、可验证的开源估价器。

## 功能

- Rust 核心计算模型，支持 P25 / P50 / P75 出价参考
- 桌面 GUI：地图选择、颜色约束、已竞出价值、手动定价、组合列表
- Windows 全局快捷键：
  - `Alt+Q` 开始计算
  - `Alt+W` 视觉扫描
  - `Alt+E` 重置条件
- 视觉扫描：直接截取主屏幕信息区域，并使用 PP-OCRv4 ONNX 识别
- 计算后自动回填唯一件数 / 格数
- 内置 4.12.2 数据，可直接构建运行

## 构建 GUI

```powershell
cargo build --release --bin bidking-analyzer
```

运行：

```powershell
.\target\release\bidking-analyzer.exe
```

也可以使用脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-gui.ps1
```

## 发布 Windows 开箱即用包

仓库提供 GitHub Actions 发布流程：

- 推送 `v*` tag 会自动构建并创建 GitHub Release
- 也可以在 Actions 页面手动运行 `Build and release`，输入 tag，例如 `v0.1.0`

Release 资产名为 `bidking-analyzer-windows-x64.zip`。解压后直接运行 `bidking-analyzer.exe`，包内已包含 OCR 所需的 PP-OCRv4 ONNX 模型和 ONNX Runtime DLL。

## 开发命令

开发 CLI 默认不编译，需要启用 `dev-cli`：

```powershell
cargo run --features dev-cli --bin bidking-dev -- calc --tier 101 --map-id 2101 --total 63 --safety 0.85 --max-show 1
```

参考输出：

```text
combos=1054 raw=766480
bid_p25=85913 bid_p50=105758 bid_p75=130632
```

测试 OCR 样本图片：

```powershell
cargo run --features dev-cli --bin bidking-dev -- ocr-image --image .\tests\fixtures\sample.png --fallback-total 63
```

实时截取主屏区域并 OCR：

```powershell
cargo run --features dev-cli --bin bidking-dev -- ocr-screen --fallback-total 63
```

## 数据与模型

- `data/auctionanalyzer-4.12.2/static_data.json`
- `data/auctionanalyzer-4.12.2/resources/MapBidCalculator.calculator_data_merged.csv`

这两个文件用于内置估价数据。

OCR 运行时文件放在 `models/ppocrv4`。为了保持 Git 仓库轻量，ONNX 模型和 ONNX Runtime DLL 不直接提交到仓库；需要视觉扫描时，按 `models/ppocrv4/README.md` 准备这些文件。

PaddleOCR / RapidOCR 模型遵循 Apache-2.0，ONNX Runtime 遵循 MIT。

## 验证

```powershell
cargo test
```

当前测试覆盖参考估价用例、4.12.2 生成用例、OCR 文本归一化和屏幕裁剪比例。

## License

本项目源码使用 MIT License。第三方模型和运行时遵循各自上游许可证。
