import traceback
from typing import List, Dict, Optional, Iterable, Tuple
from stat_lm.stat_lm import construct_model as stat_construct_model

class ModelWrapper:
    """
    Класс, который инкапсулирует всю логику генерации текста по загруженной модели и тексту.
    Тут обрабаываем подгрузку всех существующих моделей и параметров генерации под них

    load - подгрузка модели по нажатии кнопки выбора модели
    generate - генерация заданного текста текущей подгруженной моделью после команды /generate
    """
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.generate_kwargs = None

    def change_kwargs(self, value, change_str: str) -> bool:

        if self.current_model_name == 'StatLM':
            if change_str == 'temperature':
                self.generate_kwargs['generation_config'].temperature = value
                return True
            if change_str == 'sample_top_p':
                self.generate_kwargs['generation_config'].sample_top_p = value
                return True
            return False
        return False

    def load(self, model_name: str) -> bool:
        """ Load model by model_name. Return load status and error message. True if success """
        try:

            if model_name == 'StatLM':
                self.current_model_name = 'StatLM'
                self.model, self.generate_kwargs = stat_construct_model()

        except Exception as e:
            print("TRACEBACK")
            print(traceback.format_exc())
            print("*" * 20)
            return False, f"Error while loading model {model_name}: {e}"

        return True


    def generate(self, input_text: str) -> Tuple[bool, str]:
        """ generate text by context 'input_text'. Return status and message. True if success """
        if self.model is None or self.current_model_name is None:
            return False, "Need to load model"

        if not isinstance(input_text, str):
            return False, f"Inputs is not text: {type(input_text)}"

        result = self.model.generate(input_text, **self.generate_kwargs)
        if not isinstance(result, str):
            return False, f"Inference result is not string: {type(result)}"

        return True, result
