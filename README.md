# Stanford-Project
Premier League Machine learning model
MatchMind is a beginner-friendly probability project that estimates the probabilities of English Premier League (EPL) match outcomes: home win, draw, or away win.
Research Question
Can historical Premier League results help estimate the probability of a home win, draw, or away win?
Dataset
The data comes from DataHub – English Premier League.
- Training data: 2020/21–2023/24 seasons (1,520 matches)
- Test data: 2024/25 season (380 matches)
Method
MatchMind counts each outcome in the training data and calculates its historical frequency:
- Home win (H): 43.82%
- Draw (D): 22.37%
- Away win (A): 33.82%
It evaluates these probabilities on the held-out 2024/25 season using log loss. The test log loss was 1.0797.
