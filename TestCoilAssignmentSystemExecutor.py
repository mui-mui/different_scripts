###################################################################
"""
TODO: Заполнить описание
"""

__author__ = "Alexander Titov"
__version__ = "0.1"
__maintainer__ = "Alexander Titov"
__email__ = "titov_as@omk.ru"
__status__ = "Development"

###################################################################
import datetime
from functools import reduce

import java.io
from java.nio.charset import StandardCharsets
from org.apache.commons.io import IOUtils
from org.apache.nifi.processor.io import StreamCallback
from org.python.core.util import StringUtil
import traceback
import json
import os
import pickle
import random
import sys


class Observer:
    def Send(self, source):
        pass


class MessageCreator(Observer):
    message = list()
    origin = list()

    def Send(self, source):
        testHeat = source["testHeat"]
        previewHeat = source["previewHeat"]
        coilCount = source["coilCount"]
        testCoils = random.sample(list(testHeat.Coils), coilCount)

        if not not self.origin:
            for entry in self.origin:
                if entry["COIL_ID"] in [tc.Id for tc in testCoils]:
                    entry["CONTROL_UNIT"] = 1
                    self.message.append(entry)
                elif entry["COIL_ID"] in \
                        [coil.Id for coil in list(reduce(
                            lambda cur, next: cur + next,
                            [list(prHeat.Coils) for prHeat in previewHeat]) + list(testHeat.Coils))]:
                    entry["CONTROL_UNIT"] = 0
                    self.message.append(entry)


class Publisher:
    def Attach(self, observer):
        pass

    def Detach(self, observer):
        pass

    def Notify(self, source):
        pass


class Coil:
    def __init__(self):
        self.__id = None

    @property
    def Id(self):
        return self.__id

    @classmethod
    def Create(cls, _id):
        o = cls()
        o.__id = _id
        return o

    def __eq__(self, other):
        if not isinstance(other, Coil):
            return NotImplemented

        return self.Id == other.Id

    def __hash__(self):
        return hash(self.__id)


class Heat:
    def __init__(self):
        self.__id = None
        self.coils = set()

    @property
    def Id(self):
        return self.__id

    @property
    def Coils(self):
        for coil in self.coils:
            yield coil

    @property
    def CoilCount(self):
        return len(self.coils)

    @classmethod
    def Create(cls, _id):
        o = cls()
        o.__id = _id
        return o

    def IsCoilExist(self, coilId):
        return coilId in [coil.Id for coil in self.coils]

    def AddCoil(self, coil):
        self.coils.add(coil)

    def __gt__(self, other):
        return self.Id > other.Id

    def __lt__(self, other):
        return self.Id < other.Id


class Steel:
    def __init__(self):
        self.__id = None
        self.__heats = list()

    @property
    def Id(self):
        return self.__id

    @property
    def Heats(self):
        for heat in self.__heats:
            yield heat

    @property
    def HeatCount(self):
        return len(self.__heats)

    def ClearHeatsExceptLast(self):
        self.__heats = self.__heats[-1:]

    def IsHeatExist(self, headId):
        return headId in [heat.Id for heat in self.__heats]

    def AddHeat(self, heat):
        self.__heats.append(heat)

    def GetOrCreateHeat(self, headId):
        hList = list(filter(lambda item: item.Id == headId, self.__heats))
        if not hList:
            heat = Heat.Create(headId)
            self.__heats.append(heat)
        return list(filter(lambda item: item.Id == headId, self.__heats))[0]

    @classmethod
    def Create(cls, _id):
        o = cls()
        o.__id = _id
        return o


class CoilAssignmentSystemRepository:

    def __init__(self):
        self.__steels = set()

    @property
    def Steels(self):
        for steel in self.__steels:
            yield steel

    def GetOrCreateSteel(self, steelId):
        stList = list(filter(lambda item: item.Id == steelId, self.__steels))
        if not stList:
            steel = Steel.Create(steelId)
            self.__steels.add(steel)
        return list(filter(lambda item: item.Id == steelId, self.__steels))[0]

    def IsSteelExist(self, steelGrade):
        return steelGrade in [steel.Id for steel in self.__steels]


class TestCoilAssignmentSystem(Publisher):
    _observers = list()

    def __init__(self, heatCount, coilCount):
        self.__repository = CoilAssignmentSystemRepository()
        self.__heatCount = heatCount
        self.__coilCount = coilCount

    @property
    def HeatCount(self):
        return self.__heatCount

    @property
    def CoilCount(self):
        return self.__coilCount

    def AddEntry(self, entry):
        steelId = entry["STEEL_GRADE"]
        coilId = entry["COIL_ID"]

        steel = self.__repository.GetOrCreateSteel(steelId)
        heat = steel.GetOrCreateHeat(coilId[:-2])
        heat.AddCoil(Coil.Create(coilId))

        if steel.HeatCount > self.HeatCount:
            testHeat = list(steel.Heats)[-2]
            previewHeat = list(steel.Heats)[:-2]
            self.Notify({
                "testHeat": testHeat,
                "previewHeat": previewHeat,
                "coilCount": self.CoilCount,
                "modelName": entry["MODEL_NAME"]
            })
            steel.ClearHeatsExceptLast()

    def Attach(self, observer):
        self._observers.append(observer)

    def Detach(self, observer):
        raise NotImplementedError()

    def Notify(self, source):
        for observer in self._observers:
            observer.Send(source)


class TransformCallback(StreamCallback):

    def __init__(self):
        self.flowFile = None

    def process(self, inputStream, outputStream):
        try:
            # Read input FlowFile content
            input_text = IOUtils.toString(inputStream, StandardCharsets.UTF_8)
            obj = json.loads(input_text)

            testCoilAssignSys = TestCoilAssignmentSystem(
                heatCount=2,
                coilCount=1
            )
            msgCreator = MessageCreator()
            msgCreator.origin = obj
            testCoilAssignSys.Attach(msgCreator)

            for o in obj:
                testCoilAssignSys.AddEntry(o)

            outputStream.write(StringUtil.toBytes(json.dumps(msgCreator.message)))

        except:
            traceback.print_exc(file=sys.stdout)
            raise


flowFile = session.get()
if flowFile != None:
    transformCallback = TransformCallback()
    transformCallback.flowFile = flowFile
    flowFile = session.write(flowFile, TransformCallback())
    # Finish by transferring the FlowFile to an output relationship
    session.transfer(flowFile, REL_SUCCESS)
