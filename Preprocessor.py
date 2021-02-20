import base64
import pickle
import sys
import types
from abc import ABC, abstractmethod, abstractproperty

import numpy
import pandas as pd
from ast import literal_eval as make_tuple


class TypeHelper:
    """
    Расширение для проверок и преобразований
    """

    @staticmethod
    def CheckObjectType(obj, typeObj):
        if isinstance(obj, typeObj):
            return obj
        raise TypeError(f"Неверный тип объекта: '{type(obj)}', ожидался тип: '{typeObj.__name__}'")


class AbstractCheck(ABC):
    @abstractmethod
    def Check(self):
        pass


class AbstractPreprocessorDf(ABC):
    pass


class PreprocessorControrDf(AbstractPreprocessorDf):

    def __init__(self, controlDf: pd.DataFrame, foreginKey: str):
        self.__controlDf = controlDf
        self.__dfForeginKey = foreginKey
        self.__dfFeatureMinCol = None
        self.__dfFeatureMaxCol = None

    @property
    def DataFrame(self):
        return self.__controlDf

    @property
    def ForeginKeyName(self) -> str:
        return self.__dfForeginKey

    @ForeginKeyName.setter
    def ForeginKeyName(self, value) -> None:
        self.__dfForeginKey = value

    @property
    def FeatureMinColName(self) -> str:
        return self.__dfFeatureMinCol

    @FeatureMinColName.setter
    def FeatureMinColName(self, value: str) -> None:
        self.__dfFeatureMinCol = value

    @property
    def FeatureMaxColName(self) -> str:
        return self.__dfFeatureMaxCol

    @FeatureMaxColName.setter
    def FeatureMaxColName(self, value: str) -> None:
        self.__dfFeatureMaxCol = value

    @property
    def IsFeatureEquals(self) -> bool:
        return self.FeatureMaxColName == self.FeatureMinColName


class PreprocessorOriginDf(AbstractPreprocessorDf):
    def __init__(self, originDf: pd.DataFrame, dfForeginKey: str):
        self.__originDf = originDf
        self.__dfForeginKey = dfForeginKey
        self.__dfFeatureCol = None
        self.__dfFeatureMinCol = None
        self.__dfFeatureMaxCol = None

    @property
    def FeatureColumnName(self) -> str:
        return self.__dfFeatureCol

    @FeatureColumnName.setter
    def FeatureColumnName(self, value: str) -> None:
        self.__dfFeatureCol = value

    @property
    def DataFrame(self) -> pd.DataFrame:
        return self.__originDf

    @property
    def ForeginKeyName(self) -> str:
        return self.__dfForeginKey

    @property
    def IsFeatureMinMaxEquals(self) -> bool:
        return self.FeatureMaxColName == self.FeatureMinColName

    @property
    def FeatureMinColName(self):
        return self.__dfFeatureMinCol

    @property
    def FeatureMaxColName(self):
        return self.__dfFeatureMaxCol

    @FeatureMaxColName.setter
    def FeatureMaxColName(self, value: str) -> None:
        self.__dfFeatureMaxCol = value

    @FeatureMinColName.setter
    def FeatureMinColName(self, value: str) -> None:
        self.__dfFeatureMinCol = value

    @ForeginKeyName.setter
    def ForeginKeyName(self, value: str) -> None:
        self.__dfForeginKey = value

    @property
    def ControlStrategyFunc(self):
        if self.IsFeatureMinMaxEquals:
            return lambda: abs((self.__originDf[self.FeatureMaxColName] == self.__originDf[self.FeatureColumnName]) - 1)
        else:
            return lambda: abs(self.__originDf[self.FeatureColumnName].between(
                self.__originDf[self.FeatureMinColName],
                self.__originDf[self.FeatureMaxColName]
            ) - 1)

    def __add__(self, other: PreprocessorControrDf):
        self.__originDf = self.__originDf.merge(
            other.DataFrame,
            right_on=other.ForeginKeyName, left_on=self.ForeginKeyName,
            how="left"
        )
        self.__dfFeatureMinCol = other.FeatureMinColName
        self.__dfFeatureMaxCol = other.FeatureMaxColName

        return self


class CheckHelper:
    @staticmethod
    def ExecuteCheck(preprocResDf: PreprocessorOriginDf, errColName):
        df = preprocResDf.DataFrame
        errNumColName = f"{errColName}_NUMBER"
        errDescrColName = f"{errColName}_DESCRIPTION"

        if errNumColName not in df.columns:
            df[errNumColName] = 0
        if errDescrColName not in df.columns:
            df[errDescrColName] = ""

        df[f"_{errNumColName}"] = preprocResDf.ControlStrategyFunc()

        df[errNumColName] += df[f"_{errNumColName}"]
        df.loc[df[f"_{errNumColName}"] == 1, errDescrColName] += f"[{preprocResDf.FeatureColumnName} " \
                                                                 f"(min: {df[preprocResDf.FeatureMinColName].values[0]}; " \
                                                                 f"max: {df[preprocResDf.FeatureMaxColName].values[0]}; " \
                                                                 f"cur: {df[preprocResDf.FeatureColumnName].values[0]})] "


if __name__ == '__main__':
    dfsBase64Str = sys.stdin.read().split("asamm_message")
    dfs = {msg["name"]: msg["body"] for msg in [pickle.loads(base64.b64decode(msg)) for msg in dfsBase64Str]}

    dfRude = pickle.loads(base64.b64decode(dfs["checkRude"]))
    dfThech = pickle.loads(base64.b64decode(dfs["checkTechnology"]))
    dfSteelGrade = pickle.loads(base64.b64decode(dfs["checkChemestry"]))
    dfPredictionBorder = pickle.loads(base64.b64decode(dfs["checkPredictionBorder"]))
    mapperDfBag = pickle.loads(base64.b64decode(dfs["dataframe"]), encoding="bytes")

    pOriginDf = PreprocessorOriginDf(pickle.loads(mapperDfBag["resultDf"]), "COIL_OUTPUT_COIL_ID")
    pThechDf = PreprocessorControrDf(dfThech, "COIL_ID")
    pChemDf = PreprocessorControrDf(dfSteelGrade, "STEEL_GRADE_ID")
    pPredictionBorder = PreprocessorControrDf(dfPredictionBorder, "COIL_ID")

    pJoinDf = pOriginDf + pThechDf
    if not pPredictionBorder.DataFrame.empty:  # может быть пустым
        if any(pPredictionBorder.DataFrame["MAX_PREDICTION_VALUE"].isnull()):
            pPredictionBorder.DataFrame["MAX_PREDICTION_VALUE"] = numpy.iinfo(numpy.int32).max
        if any(pPredictionBorder.DataFrame["MIN_PREDICTION_VALUE"].isnull()):
            pPredictionBorder.DataFrame["MIN_PREDICTION_VALUE"] = numpy.iinfo(numpy.int32).min
        pJoinDf += pPredictionBorder
    else:
        pJoinDf.DataFrame["MAX_PREDICTION_VALUE"] = numpy.iinfo(numpy.int32).max
        pJoinDf.DataFrame["MIN_PREDICTION_VALUE"] = numpy.iinfo(numpy.int32).min

    pJoinDf.ForeginKeyName = "COIL_OUTPUT_STEEL_GRADE"
    pJoinDf += pChemDf
    errorsTypes = ("TECHNOLOGY_ERROR", "DATA_ERROR")

    """
    Проверка температур на выходе из печи (вариант проведения верефикации по полям: 
    "RHF_PROD_INFO_PYRO_TEMP_AVE", "RHF_PROD_INFO_DIS_AVE_TEMP" )
    """
    pJoinDf.FeatureMaxColName = "TF_max"
    pJoinDf.FeatureMinColName = "TF_min"
    pJoinDf.FeatureColumnName = "RHF_PROD_INFO_PYRO_TEMP_AVE"
    CheckHelper.ExecuteCheck(pJoinDf, errorsTypes[0])

    """
    Проверка толщин на выходе
    """
    pJoinDf.FeatureMaxColName = "thick_max"
    pJoinDf.FeatureMinColName = "thick_min"
    pJoinDf.FeatureColumnName = ('HSM_SC_STAT_ROLLING_EXIT_THICK', 'F6+BODY+AVG')
    CheckHelper.ExecuteCheck(pJoinDf, errorsTypes[0])

    """
    Проверка стратегии охлаждения
    """
    pJoinDf.FeatureMaxColName = "CONTROL_strategy"
    pJoinDf.FeatureMinColName = "CONTROL_strategy"
    pJoinDf.FeatureColumnName = "COOLING_REP_STRATEGY"
    CheckHelper.ExecuteCheck(pJoinDf, errorsTypes[0])

    """
    Проверка температур конца прокатки 6 пирометр 
    (проведения по двум полям: ('TEMPERATURES_BODY_AVG_VALUE', '6'), ('TEMPERATURES_TAIL_AVG_VALUE', '6')
    """
    pJoinDf.FeatureMaxColName = "FS_max"
    pJoinDf.FeatureMinColName = "FS_min"
    pJoinDf.FeatureColumnName = ('TEMPERATURES_BODY_AVG_VALUE', '6')
    CheckHelper.ExecuteCheck(pJoinDf, errorsTypes[0])

    """
    Проверка температур смотки  8 и 9 пирометр (проведения по двум полям: 
    ('TEMPERATURES.BODY_AVG_VALUE', '8'), ('TEMPERATURES.TAIL_AVG_VALUE', '8'), ('TEMPERATURES.BODY_AVG_VALUE', '9'), 
    ('TEMPERATURES.TAIL_AVG_VALUE', '9'). Сравнение будем вести по среднему на конце полосы и по 9 на середине, 
    в действующем асамме только по среднему на конце('verefication_TC_tail')
    """
    # объединение адеватные показания пирометров для проведения сверки ТК
    val = pJoinDf.DataFrame[('TEMPERATURES_BODY_AVG_VALUE', '4')].values[0]
    if val > 1500 or val < 500 or val is numpy.NAN:
        pJoinDf.DataFrame['verificate_4_and_5_pyro'] = pJoinDf.DataFrame[('TEMPERATURES_BODY_AVG_VALUE', '5')]
    else:
        pJoinDf.DataFrame['verificate_4_and_5_pyro'] = pJoinDf.DataFrame[('TEMPERATURES_BODY_AVG_VALUE', '4')]

    pJoinDf.DataFrame['verefication_TC_tail'] = (pJoinDf.DataFrame[('TEMPERATURES_TAIL_AVG_VALUE', '8')] +
                                                 pJoinDf.DataFrame[
                                                     ('TEMPERATURES_TAIL_AVG_VALUE', '9')]) / 2

    pJoinDf.FeatureMaxColName = "DC_max"
    pJoinDf.FeatureMinColName = "DC_min"
    pJoinDf.FeatureColumnName = ('TEMPERATURES_BODY_AVG_VALUE', '9')
    CheckHelper.ExecuteCheck(pJoinDf, errorsTypes[0])

    pJoinDf.FeatureColumnName = "verefication_TC_tail"
    CheckHelper.ExecuteCheck(pJoinDf, errorsTypes[0])

    """
    Проверка температур на входе в чистовую клеть 4 и 5 пирометры (вариант проведения верефикации по полям: 
    ('TEMPERATURES.BODY_AVG_VALUE', '4'),('TEMPERATURES.BODY_AVG_VALUE', '5'), ('TEMPERATURES.TAIL_AVG_VALUE', '4'), 
    ('TEMPERATURES.TAIL_AVG_VALUE', '5') исследования проведены ранее.
    """
    pJoinDf.FeatureMaxColName = "FS_in_max"
    pJoinDf.FeatureMinColName = "FS_in_min"
    pJoinDf.FeatureColumnName = "verificate_4_and_5_pyro"
    CheckHelper.ExecuteCheck(pJoinDf, errorsTypes[0])

    """
    Проверка химии. 
    Черновые фильтры пишутся по аналогии. Записи разнесены, по причине черновые фильтры являются мгновенным отказом 
    к расчету, а фильтры ТК нет, они рассчитываются если это возможно, но с пометкой....
    """
    # (minFeature, maxFeature, feature)
    featureChemistry = [
        ('MIN_C', 'MAX_C', 'Chemistry_C'),
        ('MIN_SI', 'MAX_SI', 'Chemistry_Si'),
        ('MIN_MN', 'MAX_MN', 'Chemistry_Mn'),
        ('MIN_P', 'MAX_P', 'Chemistry_P'),
        ('MIN_S', 'MAX_S', 'Chemistry_S'),
        ('MIN_CR', 'MAX_CR', 'Chemistry_Cr'),
        ('MIN_NI', 'MAX_NI', 'Chemistry_Ni'),
        ('MIN_CU', 'MAX_CU', 'Chemistry_Cu'),
        ('MIN_AL', 'MAX_AL', 'Chemistry_Al'),
        ('MIN_N', 'MAX_N', 'Chemistry_N'),
        ('MIN_MO', 'MAX_MO', 'Chemistry_Mo'),
        ('MIN_V', 'MAX_V', 'Chemistry_V'),
        ('MIN_NB', 'MAX_NB', 'Chemistry_Nb'),
        ('MIN_PB', 'MAX_PB', 'Chemistry_Pb'),
        ('MIN_TI', 'MAX_TI', 'Chemistry_Ti'),
        ('MIN_ALOXY', 'MAX_ALOXY', 'Chemistry_Aloxy'),
        ('MIN_AS', 'MAX_AS', 'Chemistry_As'),
        ('MIN_SN', 'MAX_SN', 'Chemistry_Sn'),
        ('MIN_B', 'MAX_B', 'Chemistry_B'),
        ('MIN_ZN', 'MAX_ZN', 'Chemistry_Zn'),
        ('MIN_ALSOL', 'MAX_ALSOL', 'Chemistry_Alsol')
    ]

    for struct in featureChemistry:
        pJoinDf.FeatureMaxColName = struct[1]
        pJoinDf.FeatureMinColName = struct[0]
        pJoinDf.FeatureColumnName = struct[2]
        CheckHelper.ExecuteCheck(pJoinDf, errorsTypes[0])

    """
    Грубая проверка значений полей на вхождение в указанный интервал
    """
    pJoinDf.FeatureMaxColName = "MAX_CHEKRUDE"
    pJoinDf.FeatureMinColName = "MIN_CHECKRUDE"
    for i, row in dfRude.iterrows():
        pJoinDf.DataFrame["MAX_CHEKRUDE"] = row["MAX"]
        pJoinDf.DataFrame["MIN_CHECKRUDE"] = row["MIN"]
        pJoinDf.FeatureColumnName = make_tuple(row["PROPERTY"]) if row["PROPERTY"].startswith(
            "(") else row["PROPERTY"].strip("'")
        if pJoinDf.FeatureColumnName in pJoinDf.DataFrame.columns:
            CheckHelper.ExecuteCheck(pJoinDf, errorsTypes[1])

    mapperDfBag["resultDf"] = pickle.dumps(pJoinDf.DataFrame, protocol=2)

    sys.stdout.write(base64.b64encode(pickle.dumps(mapperDfBag, protocol=2)).decode("UTF-8", errors="ignore"))
