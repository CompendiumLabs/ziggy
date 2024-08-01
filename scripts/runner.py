# patent runner script

# split files command
# split -d -a 1 --additional-suffix=.jsonl -l 2170000 ../patents_zh.jsonl patents_zh_

# run screen commands
# screen -d -m -L -Logfile screenlog.0 -S patents_zh_0 python runner.py 0
# screen -d -m -L -Logfile screenlog.1 -S patents_zh_1 python runner.py 1
# screen -d -m -L -Logfile screenlog.2 -S patents_zh_2 python runner.py 2
# screen -d -m -L -Logfile screenlog.3 -S patents_zh_3 python runner.py 3
# screen -d -m -L -Logfile screenlog.4 -S patents_zh_4 python runner.py 4
# screen -d -m -L -Logfile screenlog.5 -S patents_zh_5 python runner.py 5
# screen -d -m -L -Logfile screenlog.6 -S patents_zh_6 python runner.py 6
# screen -d -m -L -Logfile screenlog.7 -S patents_zh_7 python runner.py 7

import sys
import ziggy
from pathlib import Path

# parse arguments
index = int(sys.argv[1])
device = f'cuda:{index}'

# general paths
base_path = Path('/home/ubuntu/arizona')
data_path = base_path / 'data' / 'patents_zh'
model_path = base_path / 'hugging' / 'bge-m3'

# specific paths
input_path = data_path / 'input' / f'patents_zh_{index}.jsonl'
output_path = data_path / 'output' / f'patents_zh_{index}.torch'

# load model
print(f'Loading model: {model_path}')
embed = ziggy.HuggingfaceEmbedding(
    model_path, batch_size=32, max_len=512, device=device
)

# load data
print(f'Embedding data: {input_path}')
db = ziggy.TextDatabase.from_jsonl(
    input_path, embed=embed, name_col='appnum', text_col='text',
    device=device, qspec=ziggy.quant.Half, size=2170000
)

# save data
print(f'Saving database: {output_path}')
db.save(output_path)
