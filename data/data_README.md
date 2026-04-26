# Data

## AI Hub Korean Child Counseling Corpus

This project uses the **AI Hub Korean Child Counseling Corpus**, a publicly available dataset comprising 3,236 counseling sessions (360,816 turns, ages 7–13, 2021–2023) conducted by licensed counselors in South Korea.

### Download

The dataset is freely available at:

```
https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71680
```

Note: Registration on the AI Hub platform
is required for download.

### Dataset Statistics

| Crisis Level    | N     | Turns (mean) | Detection (%) |
|-----------------|-------|--------------|---------------|
| Normal          | 665   | 100.4        | 82.4          |
| Observation     | 634   | 106.7        | 92.9          |
| Counseling      | 621   | 108.6        | 94.5          |
| Abuse-Suspected | 644   | 117.9        | 94.4          |
| Emergency       | 672   | 123.5        | 92.4          |
| **Total**       | **3,236** | **111.5** | **91.3**  |

### Preprocessing

After downloading the raw JSON files from AI Hub, run the following preprocessing to generate the required pickle files:

**For GRU experiments** (`df_COMPLETE_for_analysis.pkl`):

```python
import json
import pandas as pd
from pathlib import Path

sessions = []
for f in Path('data/raw/').glob('*.json'):
    with open(f) as fp:
        data = json.load(fp)
    sessions.append({
        'session_id':      data['id'],
        'crisis_en':       CRISIS_MAP[data['crisis_level']],
        'has_risk':        data['has_risk'],
        'first_risk_turn': data.get('first_risk_turn'),
        'turns':           data['turns']  # list of dicts
    })

df = pd.DataFrame(sessions)
df.to_pickle('df_COMPLETE_for_analysis.pkl')
```

Crisis level mapping:
```python
CRISIS_MAP = {
    '정상군':   'Normal',
    '관찰필요': 'Observation',
    '상담필요': 'Counseling',
    '학대의심': 'Abuse-Suspected',
    '응급':     'Emergency',
}
```

**For BERT/LoRA/LLM experiments** (`full_dataset_with_dialogues_*.pkl`):

The BERT/LoRA/LLM experiments require the full dialogue texts in a compatible format. The preprocessing mirrors the above but includes text concatenation of turns. Contact the authors after the review period for preprocessing scripts.

### Ethical Considerations

- The dataset was collected with informed consent and ethical approval by the data-providing institution (AI Hub / Ministry of Science and ICT, Republic of Korea).
- All personally identifiable information was de-identified prior to public release.
- Our secondary analysis does not require additional IRB approval.
- The data must be used for research purposes only, in accordance with AI Hub terms of use.

### Citation

```bibtex
@misc{aihub2024,
  author       = {{AI Hub}},
  title        = {Korean Child Counseling Corpus},
  year         = {2024},
  howpublished = {Ministry of Science and ICT,
                  Republic of Korea},
  note         = {\url{https://aihub.or.kr/aihubdata/
                  data/view.do?dataSetSn=71680}}
}
```
