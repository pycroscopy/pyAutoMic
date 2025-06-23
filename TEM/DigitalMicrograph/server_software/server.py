import io
import numpy as np
import Pyro5.api
import DigitalMicrograph as DM

# Get the current camera and ensure it is ready for acquisition

height = 100
ds_parameters = [0, 0, 1, 1]

def beam_control():
    DM.DS_GetScanControl()
    print("beam_control_acquired")
    return

def move_beam_to(x: float=0, y: float = 0): 
    """
    Args:
        x: pixel position to go in x
        y: pixel position to go in y
    
    Return: None
    """
    global ds_parameters
    x_ds_offset, y_ds_offset, x_ds_step, y_ds_step = ds_parameters
    DM.DS_MoveBeamTo(x_ds_offset+x*x_ds_step, y_ds_offset+y*y_ds_step)
    return


def get_beam_position():
    """
    Return Beam bosition in DS coordinate system
    """
    x, y= DM.DS_GetBeamDSPosition()
    return x, y
    

def set_ds_parameter(size=512, pixelTime=30):
	rotation = 0
	lineSync = False
	paramID = DM.DS_CreateParameters(size, size, rotation, pixelTime, lineSync )
	# Add to be acquired signals
	signalIndex = 1 # 2nd assigned STEM Detector
	selectSignal = True
	dataByte = 4
	array = np.zeros(shape=(size, size), dtype='uint32')
	dmImg = DM.CreateImage(array)
	dmImg.ShowImage()
	useImgID = dmImg.GetID()
	dmImg.SetName("DS image")
	#  del dmImg # Always delete image variables again when no longer needed
	DM.DS_SetParametersSignal(paramID, signalIndex, dataByte, selectSignal, useImgID)
	return paramID, dmImg #

def get_ds_step_params(image):
    """
    Args: 
        image [can be DM.GetFrontImage() ]
    
    Returns:    
            List: [digiscan_offset_in_x, digiscan_offset_in_y, x_ds_step, y_ds_step]
    """
    im_tags = image.GetTagGroup()
    x_offset, x_ds_center, x_ds_step = 0, 0, 0
    y_offset, y_ds_center, y_ds_step = 0, 0, 0

    for group in im_tags:
        if group.__class__.__name__ == 'Py_TagGroup':
            # Check if this is the DS tag group by trying to get a known tag
            ds_tag_group_found, _ = group.GetTagAsFloat('Horizontal DS Offset')
            if ds_tag_group_found:
                _, x_offset = group.GetTagAsFloat('Horizontal DS Offset')
                _, x_ds_center = group.GetTagAsFloat('Horizontal Image Center')
                _, x_ds_step = group.GetTagAsFloat('Horizontal Spacing')
                _, y_offset = group.GetTagAsFloat('Vertical DS Offset')
                _, y_ds_step = group.GetTagAsFloat('Vertical Spacing')
                _, y_ds_center = group.GetTagAsFloat('Vertical Image Center')
                break # Terminate when the tag group is found and parameters are extracted

    x_ds_offset = x_offset - x_ds_center * x_ds_step
    y_ds_offset = y_offset - y_ds_center * y_ds_step

    return [x_ds_offset, y_ds_offset, x_ds_step, y_ds_step]


def get_scanned_image(size=512, pixelTime=10, front_image = False):
    # Create Acquisition Parameter set, front image is for DM script

    if front_image == False:
            paramID, dmImg = set_ds_parameter(size=size, pixelTime=pixelTime)
            continuous = False
            synchronous = True
            DM.DS_DialogEnabled( False )
            DM.DS_StartAcquisition( paramID, continuous, synchronous )
            DM.DS_DialogEnabled( True )
            ds_parameters = get_ds_step_params(dmImg)# here we set the parameters to the global variable
            print('ds_para', ds_parameters)
            # Remove Parameter set from memory again
            DM.DS_DeleteParameters( paramID )
            array = dmImg.GetNumArray()
            del dmImg
            return array
            
    else:
        
            dmImg = DM.GetFrontImage()
            ds_parameters = get_ds_step_params(dmImg)
            print('ds_para', ds_parameters)
            array = dmImg.GetNumArray()
            del dmImg
            return array


def activateCamera(height=200): 
	cam = DM.GetActiveCamera()
	cam.PrepareForAcquire()
	height = int(height/2)
	bin = 1
	# kproc = DM.GetCameraUnprocessedEnum()
	# kproc = DM.GetCameraGainNormalizedEnum()
    kproc = 3 # corresponds to processed spectrum 
	preImg = cam.CreateImageForAcquire( bin, bin, kproc, 1024-height, 0, 1024+height, 2048)
	return preImg, cam, kproc


def acquireCamera(cam, preImg, kproc, exposure=0.1, height=200):
	preImg.SetName( "Pre-created image container" )
	preImg.ShowImage()
	print(kproc)
	height = int(height/2)
	cam.AcquireInPlace( preImg, exposure, 1, 1, kproc, 1024-height, 0, 1024+height, 2048)
	dmImgData = preImg.GetNumArray()
	return dmImgData

def setEnergyOffset(offset = 0):
    script = f"IFSetEnergyLoss({offset}); IFWaitForFilter( )"
    DM.ExecuteScriptString(script)
    return


def closeCamera(preImg):
    """
    Always delete Py_Image variables again or memory leaks will make a problem.
    preImg: Py_Image
    Py_Image handle
    """
    del preImg        # Always delete Py_Image variables again
 

def serialize_array(array):
    array_list = array.tolist()
    dtype = str(array.dtype)
    return array_list, array.shape, dtype


@Pyro5.api.expose
class ArrayServer(object):
    """A server for returning Digiscan image and EELS acquisition to the client
    
    # setup the server and acquire image
    >>>uri = "PYRO:array.server@10.46.217.242:9094"
    >>>array_server = Pyro5.api.Proxy(uri)
    # get digiscan image
    >>>array_server.activate_camera()
    >>>array_list, shape, dtype = array_server.get_ds(front_image = False, 80)# for 80*80 image
    >>>im_array = np.array(array_list, dtype=dtype).reshape(shape)
    >>>plt.figure()
    >>>img = plt.imshow(im_array, cmap="gray")
    
    # get eels spectrum
    >>>array_server.activate_camera()#, activate the camera if not activated
    >>>array_server.acquire_camera(exposure = 0.1)# exposure in seconds
    >>>array_list, shape, dtype = array_server.get_eels()
    >>>array = np.array(array_list, dtype=dtype).reshape(shape)
    >>>plt.figure()
    >>>plt.plot(array)
    
    # get eels at a desired postion
    >>>array_server.activate_camera()#, activate the camera if not activated
    >>>array_server.set_beam_pos(x, y)
    >>>array_server.acquire_camera(exposure = 0.1)# exposure in seconds
    >>>array_list, shape, dtype = array_server.get_eels()
    >>>array = np.array(array_list, dtype=dtype).reshape(shape)
    >>>plt.figure()
    >>>plt.plot(array)
    
    """
    def get_array(self):
        """TEST FUNCTION for server fuctionality

        Returns:
            list: A list of lists representing a 1024x1024 array of random integers
            tuple: The shape of the array
            str: The data type of the array
        """
        array = np.random.randint(low=0, high=255, size=(1024, 1024), dtype=np.int16)
        array_list = array.tolist()  # Convert to a standard Python list
        dtype = str(array.dtype)
        return array_list, array.shape, dtype

    def activate_camera(self):
        """Activate the camera for acquisition
        """
        self.preImg, self.cam, self.kproc = activateCamera(height=height)
        return

    def set_energy_Offset(self, offset = 0):
        """ Set the energy offset 
        """
        setEnergyOffset(offset = offset)
        return

    def acquire_camera(self, exposure):
        """Used in eels acquisiton
        """
        self.dmImgData = acquireCamera(self.cam, self.preImg, self.kproc, exposure= exposure, height=height)
        return


    def get_eels(self, dummy = False):
        """Get eels from the camera

        Args:
            dummy (bool, optional): if True, returns random array as proxy for spectrum. Defaults to False.

        Returns:
            list: A list of lists representing a 1024x1024 array of random integers
            tuple: The shape of the array
            str: The data type of the array
        """
        if dummy == False:
            spectrum = np.sum(self.dmImgData, axis = 0)
            array_list, array_shape, dtype = serialize_array(spectrum)
        else:
            spectrum = np.random.normal(size = 144)
            array_list, array_shape, dtype = serialize_array(spectrum)
        
        return array_list, array_shape, dtype 

    def get_ds(self, size = 256, pixelTime = 10, return_params = False, front_image = False):
        image_array = get_scanned_image(size = size, pixelTime = pixelTime, front_image = front_image)
        array_list, array_shape, dtype = serialize_array(image_array)
        return array_list, array_shape, dtype 
    
    def get_front_image(self):
        image = DM.GetFrontImage()
        image_array = self.dmImg.GetNumArray()
        array_list, array_shape, dtype = serialize_array(image_array)
        return array_list, array_shape, dtype 

        
        
     
    def get_beam_control(self):
        """Returns the beam control
        """
        beam_control()
        return
        
    def get_beam_pos(self):
        """Return where is beam located

        Returns:
            _type_: _description_
        """
        x,y = get_beam_position()
        return x,y
    
                 
    def set_beam_pos(self, x, y):
        """Set beam to specific position for eels acquisition

        Args:
            x (_type_): _description_
            y (_type_): _description_
        """
        move_beam_to(x, y)
        print("beam moved to", x, y)
        return
    
    def print_ds_image_params(self, image):
        """Given an image, print the DS(digiscan) parameters
        """
        x_ds_offset, y_ds_offset, x_ds_step, y_ds_step = get_ds_step_params(image)
        print("[digiscan_offset_in_x, digiscan_offset_in_y, x_ds_step, y_ds_step]",x_ds_offset, y_ds_offset, x_ds_step, y_ds_step )
        



def main():
    host = "10.0.0.79"
    daemon = Pyro5.api.Daemon(host=host, port=9091)  
    uri = daemon.register(ArrayServer, objectId="array.server")
    print("Server is ready. Object uri =", uri)
    daemon.requestLoop()

if __name__ == "__main__":
    main()
