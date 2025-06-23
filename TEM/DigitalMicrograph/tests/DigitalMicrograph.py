class DS_ParamID:
    pass

def DS_GetScanControl():
    pass

def DS_MoveBeamTo(x, y):
    pass

def DS_GetBeamDSPosition():
    return 0, 0

def DS_CreateParameters(size, size2, rotation, pixelTime, lineSync):
    return DS_ParamID()

def DS_SetParametersSignal(paramID, signalIndex, dataByte, selectSignal, useImgID):
    pass

def DS_StartAcquisition(paramID, continuous, synchronous):
    pass

def DS_DialogEnabled(enabled):
    pass

def DS_DeleteParameters(paramID):
    pass

def GetActiveCamera():
    pass

def ExecuteScriptString(script):
    pass

class Py_Image:
    def GetNumArray(self):
        return []
    def ShowImage(self):
        pass
    def SetName(self, name):
        pass
    def GetID(self):
        return 0

def CreateImage(array):
    return Py_Image()

def GetFrontImage():
    return Py_Image()

class Py_TagGroup:
    def GetTagAsFloat(self, tag_name):
        return True, 0.0

    def __iter__(self):
        return iter([])

class DM_Image:
    def GetTagGroup(self):
        return [Py_TagGroup()]

def GetFrontImage():
    return DM_Image()

class Camera:
    def PrepareForAcquire(self):
        pass
    def CreateImageForAcquire(self, bin, bin2, kproc, height1, height2, height3, height4):
        return Py_Image()
    def AcquireInPlace(self, preImg, exposure, num1, num2, kproc, height1, height2, height3, height4):
        pass

def GetActiveCamera():
    return Camera()