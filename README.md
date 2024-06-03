# LM Assistant
Я написала статистическую модель на n-граммах и обернула ее в дружелюбного Telegram bot'а, генерирующего текст. В нем легко можно менять параметры генерации.\
Способ приближения вероятности следующего токена в модели очень прост:
$$p(x_t | x_{t-k}, x_{t-k + 1}, \ldots, x_{t - 1}) = \frac{N(x_{t-k}, x_{t-k + 1}, \ldots, x_{t - 1}, x_t) + \alpha}{N(x_{t-k}, x_{t-k + 1}, \ldots, x_{t - 1}) +  \alpha |V|}$$ Здесь $k$ - размер контекста, |V| - размер словаря, N - количество.\
Также для этой модели вручную был написан BPE токенизатор, с символом '#' для обозначения начала\конца слова.\
Обучение проходило на [датасете русской литературы](https://www.kaggle.com/datasets/d0rj3228/russian-literature/data).
## Команды
* **\start** - запуск бота
* **\help** - список всех команд
* **\params** - посмотреть текущие параметры генерации
* **\repeat** - повторить генерацию предложения (удобно, если ищешь подходящие параметры для статистической модели)
* **temperature = value** - установит температуру = value
