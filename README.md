# LM Assistant
Я взяла 2 LM модели и обернула их в дружелюбного Telegram bot'а. В нем легко можно менять параметры генерации.\
1 модель - статистическая на основе n-gram, написана вручную. Ее способ приближения вероятности следующего токена очень прост:
$$p(x_t | x_{t-k}, x_{t-k + 1}, \ldots, x_{t - 1}) = \frac{N(x_{t-k}, x_{t-k + 1}, \ldots, x_{t - 1}, x_t) + \alpha}{N(x_{t-k}, x_{t-k + 1}, \ldots, x_{t - 1}) +  \alpha |V|}$$ Здесь $k$ - размер контекста, |V| - размер словаря, N означает количество.\
Также для этой модели вручную был написан BPE токенизатор. Обучение проходило на [датасете русской литературы](https://www.kaggle.com/datasets/d0rj3228/russian-literature/data).\
2 модель - GPT2LMHeadModel из семейства GPT.
