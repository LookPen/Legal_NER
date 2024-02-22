### TODO
1. 对比实验代码
2. 动态prompt OK
3. NER和RE 跑出来的效果调整：NER有时候跑出来是0，RE 跑出来只有0.4几 OK
4. ner2 是在ner的基础上把英文标签变成中文，同时增加实体类型的提示信息

### 环境 （天杀的sentence_transformers，装完后环境蹦了）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
transformers==4.38.1
pip install datasets
pip install accelerate -U