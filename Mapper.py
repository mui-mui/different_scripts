#!/usr/bin/python3
###################################################################
"""
mapper2.0 : разбиение dataframe по строкам и создание пакета с этой
строкой с информацией какие модели к ней применить с какой моделью запускать
"""

__author__ = "Alexander Titov"
__version__ = "0.1"
__maintainer__ = "Alexander Titov"
__email__ = ""
__status__ = "Development"

###################################################################

import base64
import json
import os
import pickle
# import pyarrow
import sys
import xml.etree.ElementTree as ET
import pandas as pd
from ast import literal_eval as make_tuple

class TypeHelper:
    """
    Расширение для проверок
    """

    @staticmethod
    def CheckObjectType(obj, typeObj):
        if isinstance(obj, typeObj):
            return obj
        raise TypeError("Неверный тип объекта: '{0}', ожидался тип: '{1}'".format(type(obj), typeObj.__name__))

    @staticmethod
    def CheckFileExist(filepath):
        if os.path.isfile(filepath):
            return filepath
        raise FileNotFoundError()



class PickleHelper:
    @staticmethod
    def PickleToObj(filepath):
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        raise FileNotFoundError()


class JsonHelper:
    @staticmethod
    def Load(filepath):
        with open(filepath, 'rb') as f:
            return json.load(f)


class Model:

    def __init__(self, modelId, modelName, modelFilePath):
        self.__modelId = modelId
        self.__modelName = modelName
        self.__modelFilePath = TypeHelper.CheckFileExist(modelFilePath)
        self.__modelSettings = dict()

    @property
    def Id(self):
        return self.__modelId

    @property
    def Name(self):
        return self.__modelName

    @property
    def SteelList(self):
        for steel in self.__modelSettings["steel_grade"]:
            yield steel

    @property
    def ModelSettings(self) -> dict:
        return self.__modelSettings

    @ModelSettings.setter
    def ModelSettings(self, value: dict):
        obj = TypeHelper.CheckObjectType(value, dict)
        # распаковать кортежи
        obj["column"] = list(map(lambda item: make_tuple(item) if item.startswith(
            "(") else item, value["column"]))
        self.__modelSettings = obj

    def ToDict(self):
        return {
            "id": self.__modelId,
            "name": self.__modelName,
            "model_file": self.__modelFilePath,
            "settings": self.__modelSettings,
        }


class Mapper:
    """
    Класс описывает работу маппера.
    Содержит методы поиска подходящих моделей
    :param filePath: путь к файлу настроек
    """

    def __init__(self, filePath: str) -> None:
        self.__models = set()
        self.__Initialize(filePath)

    @property
    def ModelList(self) -> iter:
        for mod in self.__models:
            yield mod

    def FindModels(self, func) -> iter:
        """
        Поиск подходящих по атрибутам моделей
        :param func: bool Функция реализующая условие поиска атрибутов в объекте модели
        :return: Model Найденные модели
        """
        for model in filter(func, self.ModelList):
            yield model

    def __Initialize(self, filePath) -> None:
        """
        Метод для инициализации класса.
        Выполняет парсинг xml файла и заполняет список моделей
        :return:
        """
        if not os.path.isfile(filePath):
            raise FileExistsError()
        if not filePath.endswith(".xml"):
            raise Exception("File extension is not '.xml'")
        root = ET.parse(filePath)
        for mod in root.getroot().iter("model"):
            try:
                model = Model(mod.attrib.get("id"), mod.attrib.get("name"), mod.attrib.get("model"))
                model.ModelSettings = JsonHelper.Load(mod.attrib.get("settings"))
            except KeyError as err:
                raise KeyError("Setting file xml error, %s" % err)
            self.__models.add(model)


class DataBag:
    def __init__(self):
        self.model = None
        self.resultDf = None
        self.currentDf = None
        self.originDf = None
        self.steelGrade = None
        self.dfPieceId = None
        self.dfCoilId = None

    def ToDict(self):
        return {
            "model": self.model,
            "resultDf": self.resultDf,
            "currentDf": self.currentDf,
            "originDf": self.originDf,
            "steelGrade": self.steelGrade,
            "pieceId": self.dfPieceId,
            "coilId": self.dfCoilId
        }


if __name__ == '__main__':
    mainDf = pd.DataFrame(pickle.loads(base64.b64decode(sys.stdin.read())))
    mapper = Mapper(r'/scripts/Mapper/mapper-model-description-schema_3.0.xml')
    resultPack = list()
    # Разбить mainDf
    rowNum = 1
    for df in (mainDf.loc[i:i + rowNum - 1, :] for i in range(0, len(mainDf), rowNum)):
        steelGrade = df.iloc[0]["COIL_OUTPUT_STEEL_GRADE"]
        pieceId = df.iloc[0]["COIL_OUTPUT_PIECE_ID"].__str__()
        coilId = df.iloc[0]["COIL_OUTPUT_COIL_ID"].__str__()
        for m in mapper.FindModels(lambda model: steelGrade in model.SteelList):
            mObj = DataBag()
            mObj.resultDf = pickle.dumps(df, protocol=2)
            mObj.model = m.ToDict()
            mObj.steelGrade = steelGrade
            mObj.dfPieceId = pieceId
            mObj.dfCoilId = coilId
            resultPack.append(mObj.ToDict())

    sys.stdout.write(base64.b64encode(pickle.dumps(resultPack, protocol=2)).decode("UTF-8", errors="ignore"))
