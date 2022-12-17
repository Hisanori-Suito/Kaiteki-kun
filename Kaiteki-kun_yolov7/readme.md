# 快適くん-YOLOv7

## IPアドレスの準備

1.  接続するSpresenseと繋がっているwifiルータのIPアドレスを取得(例：Linuxではコマンド`ifconfig`)
2.  取得したアドレスを`detect.py`の63行目`"http://***.***.*.**/?test="`(1台目)、67行目`"http://***.***.*.**/?test="`(2台目)に記入
    (同じアドレスを記入した場合は1台での動作となる)

## 学習済みのモデルのダウンロード

- 次のコマンドで学習済みのモデルをダウンロード
    ```
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt﻿
    ```

## 検証

1.  次のコマンドで検証を開始する
    ```bash
    python detect.py --source inference/spresense/ --weights yolov7-e6e.pt --conf 0.25 --img-size 1280 --device 0﻿
    ```

   - Spresenseの画像は`inference/spresense/`内に1台ずつ交互に保存される
     (保存できない場合はスキップされる)
   - 物体検出を行った結果の画像は`runs/detect/exp**/`内に保存される

2.  検証が終わったら`ctrl` + `z`(Linuxの場合)などのプログラムを停止させるコマンドで検証を停止させる
3.  必要があれば`ctrl` + `d`(Linuxの場合)などのプログラムを終了させるコマンドでプログラムの実行を完全に終了させる

## 可変パラメータ

1.  忘れ物検出
   - `detect.py`の166,169,202行目及び`utils/plots.py`の72行目 `"******"` に記入するラベル名を変えることで忘れ物検出したい物を設定することができる(初期handbag)

2.  混雑検出
   - `detect.py`の197,202行目 `crowd_1(2) - crowd_2(1) > *` に記入する数値を変えることで警告音を鳴らす人数差を設定することができる(初期値1)

## 補注

- 今回のプログラムの大半はYOLOv7をそのまま使用しているため、必要に応じて公式の[git](https://github.com/WongKinYiu/yolov7)や[論文](https://arxiv.org/abs/2207.02696)を参考にすると良い

