# PP-OCRv4 ONNX Models

This directory is the expected location for the optional OCR runtime assets used
by the GUI visual scan feature. The large model/runtime binaries are ignored by
Git and should be downloaded or copied here locally.

- `ch_PP-OCRv4_det_infer.onnx`
- `ch_ppocr_mobile_v2.0_cls_infer.onnx`
- `ch_PP-OCRv4_rec_infer.onnx`
- `onnxruntime.dll`
- `onnxruntime_providers_shared.dll`

The Rust code loads this directory automatically from `models/ppocrv4`, or from
the path specified by `BIDKING_PPOCRV4_DIR`.

The PaddleOCR models are from the PaddleOCR/RapidOCR ecosystem and are licensed
under Apache-2.0. ONNX Runtime is licensed under MIT.
