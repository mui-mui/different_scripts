import base64
import datetime
import importlib.util
import inspect
import pickle
import sys
from abc import ABC, abstractmethod
from typing import Optional, Any
import pandas
import pandas as pd
import torch
from torch import nn
from torch.functional import F
from sklearn.metrics import r2_score, mean_squared_error
from joblib import dump, load
from torch.utils.data import Dataset, DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')


class Message:
    """
    Класс представляет объект входящего сообщения
    от предыдущего блока
    """

    def __init__(self):
        self.model = None
        self.resultDf = None
        self.originDf = None
        self.currentDf = None

    @classmethod
    def Create(cls, model: dict, resultDf: Optional[pandas.DataFrame], originDf: Optional[pandas.DataFrame],
               currentDf: Optional[pandas.DataFrame]):
        o = cls()
        o.model = model
        o.resultDf = resultDf
        o.originDf = originDf
        o.currentDf = currentDf
        return o


class ModuleLoaderHelper:
    """
    Класс для управления внешними модулями системы
    """

    @staticmethod
    def LoadModule(moduleName, filePath):
        spec = importlib.util.spec_from_file_location(moduleName, filePath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return Module(mod, spec)

    @staticmethod
    def ReloadModule(moduleObj):
        if isinstance(moduleObj, Module):
            moduleObj.Scpec.loader.exec_module(moduleObj.Module)
            return
        raise TypeError("Аргумент не является типом 'Module'")


class Module:
    """
    Класс контейнер для хранения загруженного модуля
    и его спецификации
    """

    def __init__(self, moduleObj, specObj):
        self.__moduleObj = self.__CheckModuleType(moduleObj)
        self.__specObj = specObj

    @property
    def Module(self):
        return self.__moduleObj

    @property
    def Scpec(self):
        return self.__specObj

    def __CheckModuleType(self, moduleObj):
        if inspect.ismodule(moduleObj):
            return moduleObj
        raise AttributeError()

    def __str__(self):
        strBuild = f"\n-------------\n"
        strBuild += "Объект модуля\n"
        strBuild += f"Объект {type(self.__moduleObj)}\n"
        strBuild += f"Спецификация {type(self.__specObj)}\n"
        return strBuild


class ModelHandler(ABC):
    """
    Интерфейс Обработчика объявляет метод построения цепочки обработчиков модели. Он
    также объявляет метод для выполнения запроса.
    """

    @abstractmethod
    def SetNextHandler(self, handler):
        pass

    @abstractmethod
    def Handle(self, modelDataObj: Message):
        pass

    @abstractmethod
    def SetExceptionModelId(self, *models):
        """
        Список ид моделей, для которых должет применятся текущий обработчик
        :param models:
        :return:
        """
        pass


class AbstractModelHandler(ModelHandler):
    _nextHandler: ModelHandler = None
    _exceptionModelsForCurrentHandler = set()

    def SetNextHandler(self, handler: ModelHandler) -> ModelHandler:
        self._nextHandler = handler
        return handler

    @abstractmethod
    def Handle(self, modelDataObj):
        if self._nextHandler:
            return self._nextHandler.Handle(modelDataObj)
        return None

    def SetExceptionModelId(self, *models):
        """
        Для каждого обработчика можно определить список ид моделей
        с которыми текущий обработчик не будет работать
        :param models:
        :return:
        """
        self._exceptionModelsForCurrentHandler = set(list(models))


class ExistColumnHandler(AbstractModelHandler):
    """
    Обработчик проверяет существование всех необходимых.
    колонок в df необходимых для запуска модели
    """

    def Handle(self, modelDataObj):
        diffCols = set(modelDataObj.model["settings"]["column"]) - set(modelDataObj.originDf.columns)
        if not not diffCols:
            modelDataObj.resultDf[
                "WORK_MODEL_RESULT_STATUS"] = 1  # 1 - модель не прошла из-за ошибок грубой проверки
            modelDataObj.resultDf[
                "WORK_MODEL_RESULT_DESCRIPTION"] = f"Error: not found model columns in dataframe: " \
                                                   f"[{'; '.join([col.__str__() for col in diffCols])}]"
            return modelDataObj
        return super().Handle(modelDataObj)


class RudeErrorHandler(AbstractModelHandler):
    """
    Обработчик проверяет входящий df на наличие ошибкок.
    df ошибками исключается из цепочи обработчиков
    """

    def Handle(self, modelDataObj):
        if modelDataObj.model["id"] not in super()._exceptionModelsForCurrentHandler:
            if modelDataObj.resultDf["DATA_ERROR_NUMBER"].values[0] > 0:
                modelDataObj.resultDf[
                    "WORK_MODEL_RESULT_STATUS"] = 1  # 1 - модель не прошла из-за ошибок грубой проверки
                return modelDataObj
        return super().Handle(modelDataObj)


class ColumnsHandler(AbstractModelHandler):
    """
    Обработчик проверяет фильтрует и сортирует порядок df.
    """

    def Handle(self, modelDataObj):
        if modelDataObj.model["id"] not in super()._exceptionModelsForCurrentHandler:
            colOrderList = list(modelDataObj.model["settings"]["column"])
            df = modelDataObj.currentDf.drop(modelDataObj.currentDf.columns.difference(colOrderList), axis=1)
            df = df.dropna()
            modelDataObj.currentDf = df[colOrderList]
        return super().Handle(modelDataObj)


class CategoricalFeaturesHandler(AbstractModelHandler):
    """
    Обработчик проводит замену категориальных переменных по словарю.
    """

    def Handle(self, modelDataObj):
        if modelDataObj.model["id"] not in super()._exceptionModelsForCurrentHandler:
            modelDataObj.currentDf["COOLING_REP_STRATEGY"] = modelDataObj.currentDf["COOLING_REP_STRATEGY"].map(
                modelDataObj.model["settings"]["encoder"])
        return super().Handle(modelDataObj)


class NormalizationHandler(AbstractModelHandler):
    """
    Обработчик проверяет фильтрует и сортирует порядок df.
    """

    def Handle(self, modelDataObj):
        if modelDataObj.model["id"] not in super()._exceptionModelsForCurrentHandler:
            sc = load(modelDataObj.model["settings"]["scaler_file"])
            modelDataObj.currentDf = sc.transform(modelDataObj.currentDf)
        return super().Handle(modelDataObj)


class BeforePredictHandler(AbstractModelHandler):
    """
    Обработчик для перобразования данных к виду загрузки для дальнейшего запуска предсказания
    """

    def Handle(self, modelDataObj):
        if modelDataObj.model["id"] not in super()._exceptionModelsForCurrentHandler:
            modelDataObj.currentDf = torch.from_numpy(modelDataObj.currentDf).float()
            modelDataObj.currentDf = TensorDataset(modelDataObj.currentDf)
            modelDataObj.currentDf = DataLoader(modelDataObj.currentDf, batch_size=len(modelDataObj.currentDf),
                                                shuffle=False)
        return super().Handle(modelDataObj)


class ModelCalculation:
    def __init__(self, modelDataObj: Message):
        self.__modelDataObj = modelDataObj

    def Execute(self):
        # создание звеньев цепи
        existColumnHandler = ExistColumnHandler()
        rudeErrorHandler = RudeErrorHandler()
        columnsHandler = ColumnsHandler()
        categoricalFeaturesHandler = CategoricalFeaturesHandler()
        normalizationHandler = NormalizationHandler()
        beforePredictHandler = BeforePredictHandler()

        # цепочка вызовов обработчиков
        existColumnHandler \
            .SetNextHandler(rudeErrorHandler) \
            .SetNextHandler(columnsHandler) \
            .SetNextHandler(categoricalFeaturesHandler) \
            .SetNextHandler(normalizationHandler) \
            .SetNextHandler(beforePredictHandler)
        # запуск цепи
        existColumnHandler.Handle(self.__modelDataObj)

        self.__ModelPredict()

        return

    def __CheckPredictBorder(self):
        self.__modelDataObj.resultDf["WORK_MODEL_RESULT_STATUS"] = \
            abs(self.__modelDataObj.resultDf["PREDICTION_MODEL_RESULT"].between(
                self.__modelDataObj.resultDf["MIN_PREDICTION_VALUE"],
                self.__modelDataObj.resultDf["MAX_PREDICTION_VALUE"]
            ) - 1)

    def __ModelPredict(self):
        self.__modelDataObj.resultDf["PREDICTION_MODEL_RESULT"] = -1
        statusResult = self.__modelDataObj.resultDf["WORK_MODEL_RESULT_STATUS"].values[0]
        if statusResult != 1:  # запустить модель, только если данные прошли грубую проверку на ошибки (DATA_ERROR_NUMBER = 0)
            try:
                # создание объекта класса текущей модели
                module = ModuleLoaderHelper.LoadModule(
                    moduleName=self.__modelDataObj.model["id"],
                    filePath=self.__modelDataObj.model["model_file"]
                ).Module
                device = torch.device('cpu')
                model = module.NeuralModel()
                model.load_state_dict(
                    torch.load(self.__modelDataObj.model["settings"]["tensor_file"], map_location=device))
                model.zero_grad()
                model.eval()

                with torch.no_grad():
                    for x in self.__modelDataObj.currentDf:
                        predictResult = model(x[0])

                self.__modelDataObj.resultDf["PREDICTION_MODEL_RESULT"] = \
                    predictResult.numpy()

                # проверить вхождение модели в интервал
                self.__CheckPredictBorder()

            except Exception as e:
                self.__modelDataObj.resultDf["WORK_MODEL_RESULT_STATUS"] = -1
                self.__modelDataObj.resultDf["WORK_MODEL_RESULT_DESCRIPTION"] = f"Fail: {e}"


if __name__ == '__main__':
    preprocDfBag = pickle.loads(base64.b64decode(sys.stdin.read()), encoding="bytes")
    preprocDf = pickle.loads(preprocDfBag["resultDf"]).set_index('COIL_OUTPUT_COIL_ID')

    # Инициализация df колонками генерируемыми запуском модели
    preprocDf["PREDICTION_MODEL_RESULT"] = -1
    preprocDf["WORK_MODEL_RESULT_DESCRIPTION"] = ""
    preprocDf["WORK_MODEL_RESULT_STATUS"] = -1
    preprocDf["MODEL_NAME"] = preprocDfBag["model"]["id"]
    preprocDf["MODEL_DESCRIPTION"] = preprocDfBag["model"]["name"]
    preprocDf["STEEL_GRADE"] = preprocDfBag["steelGrade"]
    preprocDf["PREDICTION_DATETIME"] = (datetime.datetime.utcnow().replace(microsecond=0)) #datetime.datetime.now().strftime('%d-%m-%y %H:%M:%S')

    mObj = Message.Create(
        model=preprocDfBag["model"],
        # формирование таблицы асамм
        resultDf=preprocDf[["TECHNOLOGY_ERROR_NUMBER",
                            "TECHNOLOGY_ERROR_DESCRIPTION",
                            "DATA_ERROR_NUMBER",
                            "DATA_ERROR_DESCRIPTION",
                            "WORK_MODEL_RESULT_DESCRIPTION",
                            "WORK_MODEL_RESULT_STATUS",
                            "PREDICTION_MODEL_RESULT",
                            "MODEL_NAME",
                            "MODEL_DESCRIPTION",
                            "STEEL_GRADE",
                            "MAX_PREDICTION_VALUE",
                            "MIN_PREDICTION_VALUE",
                            "PREDICTION_DATETIME"]].copy(),
        originDf=preprocDf.copy(),
        currentDf=preprocDf.copy()
    )

    modelCalculation = ModelCalculation(mObj)
    modelCalculation.Execute()

    preprocDfBag["resultDf"] = pickle.dumps(mObj.resultDf, protocol=2)
    preprocDf["originDf"] = None
    preprocDf["current"] = None

    sys.stdout.write(base64.b64encode(pickle.dumps(preprocDfBag, protocol=2)).decode("UTF-8", errors="ignore"))
