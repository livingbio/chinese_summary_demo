# A Demo Site for English Summarization

## Installation

```shell-script
git clone git@github.com:livingbio/chinese_summary_demo.git
cd chinese_summary_demo
virtualenv venv
source venv/bin/activate

pip install scipy
pip install -r requirements.txt

cd src
cp chinese_summary_demo/settings/local.sample.env chinese_summary_demo/settings/local.env
vi chinese_summary_demo/settings/local.env  # replace the SECRET_KEY
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

## Test Data

[《華盛頓郵報》社論指史諾登不應獲特赦 遭批評忘恩負義](https://gist.github.com/banyh/7268cc49a350f52546f15fb766945b0a)
[愛因斯坦：只有兩樣東西可能沒有極限，其一是宇宙，其二是人類的愚蠢](https://gist.github.com/banyh/93e8876ae93063cca65f2684d3e80c9a)
[唐鳳首度VR技術跨國連線 小學生好奇提問性別](https://gist.github.com/banyh/021d7ddc11b50b7677f7e72c2366bff7)
[背包客有福了！Google推出免上網也能自動規劃的Google Trips](https://gist.github.com/banyh/8084f163bf8701036ef0156236ed0641)
[無懼風雨！布做外殼的可愛風電動小車 rimOnO](https://gist.github.com/banyh/cd962fccc55f1c0c0a7714e36556b3e9)
[神貼片招喚「自體止痛機制」！打到「痛點」 Livia 募標爆衝1340%](https://gist.github.com/banyh/42bd39e316345ea8d2ac269e10d7a149)
[小偷剋星！Sherlock 實現自行車的「劫盜地圖」](https://gist.github.com/banyh/705774d05f817f020f94ed0676d9436a)
[隔空觸控！AirBar 讓筆電變身魔法畫布](https://gist.github.com/banyh/395a8bfcedc6024b7b9f4e84aead0cef)
