# Author credits - Utkarsh Pratiush <utkarshp1161@gmail.com>

import logging
import Pyro5.api
from autoscript_tem_microscope_client import TemMicroscopeClient
from autoscript_tem_microscope_client.enumerations import DetectorType, CameraType, OptiStemMethod, OpticalMode
from autoscript_tem_microscope_client.structures import RunOptiStemSettings, RunStemAutoFocusSettings, Point, StagePosition, AdornedImage
from typing import Tuple, Dict, List, Optional, Union
import numpy as np
from datetime import datetime
import Pyro5
import copy
from stemOrchestrator.simulation import DMtwin


class TFacquisition:
    # acquires HAADF[with scalebar], CBED, EDX
    # also does optistem(C1_A1, C1, B2_A2)
    # later - drift correct on HAADF, defocus series, Segment regions and cbed
    def __init__(self, microscope: TemMicroscopeClient, offline: bool = True) -> None:
        self.offline = offline
        """Initialize the microscope and set up acquisition parameters."""
        logging.info("Starting microscope initialization...")
        self.microscope = microscope

        
        try:
            self.ceta_cam = self.microscope.detectors.get_camera_detector(CameraType.BM_CETA)
            logging.info("CETA camera initialized")
            self.haadf_det = self.microscope.detectors.get_scanning_detector(DetectorType.HAADF)
            logging.info("HAADF detector initialized")
        except Exception as e:
            logging.error(f"Failed to initialize detectors: {str(e)}")
            raise
        logging.info("LOGGING STATE of the Microscope================================================")
        self.query_state_of_microscope()
        # Log early the state of the microscope [stage{x, y, z, A, B}, paused beam postion{b/w 0 and 1 in x and y}, screen_current]
        # self.query_vacuum_valves()
        # self.current_stage_position = self.query_stage_position()
        # self.last_stage_postion = copy.deepcopy(self.current_stage_position)
        # self.initial_stage_postion = copy.deepcopy(self.current_stage_position)
        
        # self.last_paused_beam_positon = self.query_paused_beam_positon()
        # self.current_paused_beam_positon = copy.deepcopy(self.last_paused_beam_positon) 
        # self.initial_paused_beam_position = copy.deepcopy(self.last_paused_beam_positon) 

        # self.last_beam_shift_pos = self.query_beam_shift_position()
        # self.initial_beam_shift_pos = copy.deepcopy(self.last_beam_shift_pos)
        # self.current_beam_shift_pos = copy.deepcopy(self.last_beam_shift_pos)
        

        # self.state = self.microscope.state.__dir__()
        # logging.info(f"THE AS provided state: microscope.state - {self.state}")
        
        logging.info("Microscope initialization completed successfully")
  

    def query_state_of_microscope(self) -> None:
        """
        vacuum, optical mode, is beam_blanked, 
        beam shift position, paused beam postion, stage position,
        FOV, screen current, 
        """
        logging.info("Request to log the state of the microscope")
        self.query_list_of_detectors()
        self.query_vacuum_valves()
        self.query_beam_shift_position()
        self.query_is_beam_blanked()
        self.query_optical_mode()
        self.query_haadf_state()
        self.query_ceta_state()
        # self.query_FOV() --> only present in STEM mode
        self.query_screen_postion()
        self.query_screen_current()
        self.query_paused_beam_positon()
        self.query_stage_position()
        logging.info("DONE: quering the state of microscope")
        pass
        

    def query_relevant_metadata_of_image(self, image: AdornedImage) -> None:
        logging.info("Request to query the metadata of an AdornedImage object")

        try:
            logging.info(f"Defocus: {image.metadata.optics.defocus}")
            logging.info(f"C2 Lens Intensity: {image.metadata.optics.c2_lens_intensity}")
            logging.info(f"Beam Tilt: {image.metadata.optics.beam_tilt}")
            logging.info(f"Stage Position: {image.metadata.stage_settings.stage_position}")
        except AttributeError as e:
            logging.error(f"Error accessing metadata attributes: {e}")

        logging.info("DONE: Request to query the metadata of an AdornedImage object")
    
    def template_func(self) -> str:
        logging.info("Request to query the ....")
        val = "bla"
        logging.info(f"DONE: {val} ")
        return val
        
    def query_list_of_detectors(self) -> List:
        logging.info("Request to query the list of detectors on the Scope")
        val = self.microscope.detectors.camera_detectors
        logging.info(f"DONE: querying the list of detectors: {val} ")
        return val
    
    def query_optical_mode(self) -> str:
        logging.info("Request to query the optical mode of the instrument")
        optical_mode = self.microscope.optics.optical_mode
        logging.info(f"DONE: the optical mde of the microsocpe is {optical_mode}")
        return optical_mode
    
    def set_to_STEM_mode(self) -> None:
        logging.info("Request to set to Optical mode =  STEM mode")
        self.microscope.optics.optical_mode = OpticalMode.STEM
        logging.info(f"DONE: set to Optical mode =  STEM mode")
        return None     
    
    def set_to_TEM_mode(self) -> None:
        logging.info("Request to set to Optical mode =  TEM mode")
        self.microscope.optics.optical_mode = OpticalMode.TEM
        logging.info(f"DONE: set to Optical mode =  TEM mode")
        return None   
    
    def query_screen_postion(self) -> str:
        logging.info("Request to query the position of the screen")
        val = self.microscope.detectors.screen.position
        logging.info(f"DONE: The state of the screen is {val} ")
        return val

    def query_haadf_state(self) -> str:
        logging.info("Request to query the position of the HAADF")
        val = self.haadf_det.insertion_state
        logging.info(f"DONE: The state of the HAADF is {val} ")
        return val
    
    def query_ceta_state(self) -> str:
        logging.info("Request to query the position of the CETA")
        val = self.ceta_cam.insertion_state
        logging.info(f"DONE: The state of the CETA is {val} ")
        return val

    def acquire_haadf(self, exposure: float = 40e-9, resolution: int = 512, return_pixel_size = False) -> Tuple[np.ndarray, str, Optional[Tuple]]:
        """Acquire HAADF image."""
        logging.info("Acquiring HAADF image.")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.microscope.optics.unblank()
        image = self.microscope.acquisition.acquire_stem_image(DetectorType.HAADF, resolution, exposure)# takes 40 seconds
        self.microscope.optics.blank()
        image.save(f"HAADF_image_{current_time}")# saves the tiff
        haadf_tiff_name = f"HAADF_image_{current_time}.tiff"
        logging.info("saving HAADF image as TF which has all the metadata..also returning an array")
        # convert the image to noarray and return that as well
        img = image.data - np.min(image.data)
        image_data = (255*(img/np.max(img))).astype(np.uint8)
        # HAADF_tiff_to_png(f"HAADF_image_{current_time}.tff")
        logging.info("Done: Acquiring HAADF image - beam is blanked after acquisition - HAADF det is inserted")

        if return_pixel_size:
            pixel_size_tuple = image.metadata.binary_result.pixel_size.x, image.metadata.binary_result.pixel_size.y
            return image_data, haadf_tiff_name, pixel_size_tuple
        
        return image_data, haadf_tiff_name

    def acquire_ceta(self, exposure: float = 0.1, resolution: int = 4096) -> Tuple[np.ndarray, str, Optional[Tuple]]:
        """Acquire CETA image.
        Args:
            exposure : float : 0.2 means 0.2 seconds i.e 200ms
        
        """
        logging.info("Acquiring CETA image.")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ceta_cam.insert()
        self.microscope.optics.unblank()
        image = self.microscope.acquisition.acquire_stem_image(CameraType.BM_CETA, resolution, exposure)# takes 40 seconds
        self.microscope.optics.blank()
        self.ceta_cam.retract()

        image.save(f"CETA_image_{current_time}")# saves the tiff
        ceta_tiff_name = f"CETA_image_{current_time}.tiff"
        logging.info("saving CETA image as TF which has all the metadata..also returning an array")
        # convert the image to noarray and return that as well
        img = image.data - np.min(image.data)
        image_data = (255*(img/np.max(img))).astype(np.uint8)
        # n = ceta_image_data.shape[0]# 4096
        # center_half = ceta_image_data[n // 4: 3 * n // 4, n // 4: 3 * n // 4]
        # center_quarter = ceta_image_data[n // 2: 3 * n // 4, n // 2: 3 * n // 4]
        # center_quarter = ceta_image_data[1024:-1024, 1024:-1024]

        # CETA_tiff_to_png(f"CETA_image_{current_time}.tff")
        logging.info("Done: Acquiring CETA image. Beam is blanked and ceta detector is retracted")
        pixel_size_tuple = image.metadata.binary_result.pixel_size.x, image.metadata.binary_result.pixel_size.y

    
        return image_data, ceta_tiff_name
            
    def drift_correct(self, haadf_image: np.ndarray, haadf_image_shifted: np.ndarray):
        """Perform drift correction based on HAADF."""
        logging.info("Performing drift correction.")
        # Simulated drift correction
        return "corrected drift"

    def get_edge_coordinates(self, corrected_haadf):
        """Extract edge coordinates from HAADF image."""
        logging.info("Extracting edge coordinates.")
        # Simulated edge detection
        return [(100, 200), (150, 250)]

    def get_cbed_at_center(self) -> np.ndarray:
        """Get cbed pattern at center."""
        logging.info("Acquiring ceta ")
        # Simulated edge detection
        return 
    
    
    def acquire_edx(self, edge_coords):
        """Perform EDX at extracted edge coordinates."""
        logging.info(f"Performing EELS at {edge_coords}.")
        # Simulated EELS acquisition
        return "eels_spectrum"

    def autofocus(self, exposure : float = 1e-5, haadf_resolution: int = 1024) -> None:
        """Perform autofocus with HAADF detector"""
        logging.info(f"Performing autofocus with HAADF detector")
        settings = RunStemAutoFocusSettings(DetectorType.HAADF, haadf_resolution, exposure, True, reset_beam_blanker_afterwards=True)
        self.microscope.auto_functions.run_stem_auto_focus(settings)
        logging.info(f"DONE autofocus with HAADF detector")
        return None

        
    def optistem_c1(self, dwell_time : float = 2e-06, cutoff_in_pixels: int = 5) -> None:
        """Perform optistem - defocus - c1"""
        logging.info(f"Performing c1 optistem correction.")
        settings = RunOptiStemSettings(method=OptiStemMethod.C1, dwell_time=dwell_time, cutoff_in_pixels=cutoff_in_pixels)
        self.microscope.auto_functions.run_opti_stem(settings)
        logging.info(f"DONE c1 optistem correction.")
        return None

    def optistem_c1_a1(self, dwell_time : float = 2e-06, cutoff_in_pixels: int = 5) -> None:
        """Perform optistem - defocus - c1_a1"""
        logging.info(f"Performing c1_a1 optistem correction.")
        settings = RunOptiStemSettings(method=OptiStemMethod.C1_A1, dwell_time=dwell_time, cutoff_in_pixels=cutoff_in_pixels)
        self.microscope.auto_functions.run_opti_stem(settings)
        logging.info(f"DONE c1 a1 optistem correction.")
        return None
    
    def optistem_b2_a2(self, dwell_time: float = 2e-06) -> None:
        """Perform optistem - 2nd order"""
        logging.info(f"Performing B2_A2 optistem correction.")
        settings = RunOptiStemSettings(method=OptiStemMethod.B2_A2, dwell_time=dwell_time)
        self.microscope.auto_functions.run_opti_stem(settings)
        logging.info(f"DONE B2_A2 optistem correction.")
        return None

    def query_paused_beam_positon(self) -> Point:
        # x lies b/w 0 to 1 and y lies b/w 0 to 1
        logging.info("Request to query the paused beam postion")
        pos = self.microscope.optics.paused_scan_beam_position
        logging.info(f"DONE: Query Paused beam position: which is at {pos}")
        return pos

    def move_paused_beam(self, x : float, y: float ) -> None:
        # x lies b/w 0 to 1 and y lies b/w 0 to 1
        logging.info(f"Set beam position: old {self.microscope.optics.paused_scan_beam_position}")
        self.microscope.optics.paused_scan_beam_position = [x, y]
        logging.info(f"UPDATED beam position: New{self.microscope.optics.paused_scan_beam_position}")
        return

    def rad2miliDegree(self, radians : float) -> float:
        radians_to_degree_value = np.rad2deg(radians)*(10**-3)
        return radians_to_degree_value
    

    def rad2miliDegree_stage(self, alpha_beta_rad: Tuple[float, float], single_tilt_holder = True) -> Tuple[float, float]:
        # TODO: handle the holder better --> set variable while initalizing
        if single_tilt_holder:
            return self.rad2miliDegree(alpha_beta_rad[0])
        
        else:
            alpha, beta = (self.rad2miliDegree(i) for i in alpha_beta_rad)  # Unpack to ensure exactly two elements
        return (alpha, beta)
    
    def miliDegree2rad(self, millidegrees: float) -> float:
        """Convert millidegrees back to radians."""
        return np.deg2rad(millidegrees * (10**3))

    def miliDegree2rad_stage(self, alpha_beta_milideg: Tuple[float, float]) -> Tuple[float, float]:
        """Convert a tuple of millidegrees back to a tuple of radians."""
        alpha, beta = (self.miliDegree2rad(i) for i in alpha_beta_milideg)  # Unpack to ensure exactly two elements
        return (alpha, beta)
    
    def query_stage_position(self) -> StagePosition:
        #Returns the stage position (X, Y, Z) in meters, A and B in radians.
        logging.info(f"Querying the stage position now: the stage position (X, Y, Z) in meters, A and B in radians")
        pos = self.microscope.specimen.stage.position
        pos_dict_m = {"x" : pos[0], "y" : pos[1], "z" : pos[2]}
        pos_dict_nm = {"x" : pos[0]*10**(-9), "y" : pos[1]*10**(-9), "z" : pos[2]*10**(-9)}

        angle_tuple_rad = (pos[3], pos[4])# A and B
        angle_list_miliDegree = self.rad2miliDegree_stage(angle_tuple_rad)
        
        logging.info(f"Query stage position: which is at {pos}, i.e pos in nm {pos_dict_nm}, angles in mili degree{angle_list_miliDegree}")
        return pos

    def undo_last_stage_movement(self) -> None:
        logging.info("Request to undo last stage movement")
        pos = self.last_stage_postion
        self.microscope.specimen.stage.absolute_move(pos)
        self.current_stage_position = self.query_stage_position()
        logging.info(f"DONE : undo last stage movement - stage is now at {self.current_stage_position}")
        return None

    def move_stage_translation_relative(self, x: float, y: float, z: float, relative: bool = True) ->  None:
        ## put x, y, z in nm
        logging.info("Request to translate the stage relative values")
        self.microscope.specimen.stage.relative_move(StagePosition(x = x*10**-9, y = y*10**-9 , z = z*10**-9))
        logging.info("Done: translate the stage relative values")

        pass     

    def move_stage_translation_absolute(self, x: float, y: float, z: float, relative: bool = True) ->  None:
        ## put x, y, z in m - unlike the  move_stage_translation_relative
        logging.info("Request to translate the stage absolute values")
        self.microscope.specimen.stage.absolute_move_safe(StagePosition(x = x, y = y , z = z))
        logging.info("DONE: shifted atage to absolute values")
        pass     
        
    def move_stage_rotation(self, alpha: float = None, beta: float = None, single_tilt_holder = True):
        ## put A and B in milidegrees 
        # Single tilt holder may have just one A angle
        logging.info("Request to rotate the stage")
        angles_in_radians = self.miliDegree2rad_stage((alpha, beta))
        if single_tilt_holder:
            logging.info("holder is sigle Tilt")
            self.microscope.specimen.stage.relative_move(StagePosition(a = angles_in_radians[0]))
            logging.info(f"Stage rotated by {alpha} milidegree")
        else:
            logging.info("holder is Double Tilt")
            self.microscope.specimen.stage.relative_move(StagePosition(a = angles_in_radians[0], b = angles_in_radians[1]))
            logging.info(f"Stage rotated by {alpha, beta} milidegree")
        pass
    
    def query_beam_shift_position(self) -> Point:
        # the pos we get here is in m
        logging.info("Request to query the beam(not paused beam) position")
        pos = self.microscope.optics.deflectors.beam_shift
        logging.info(f"Done to query the beam(not paused beam) position: at {pos}")
        return pos
    
    def move_beam_shift_positon(self, pos = Tuple[float, float])-> None:
        # pos is in metres for x and y
        self.last_beam_shift_pos = self.microscope.optics.deflectors.beam_shift
        logging.info(f"Request to shift the beam(not paused beam), old position in meters {self.last_beam_shift_pos}")
        self.microscope.optics.deflectors.beam_shift = list(pos)
        self.current_beam_shift_pos = self.microscope.optics.deflectors.beam_shift
        logging.info(f"Done: beam shift(not paused beam) to {self.current_beam_shift_pos}")
        return 
        
    def query_is_beam_blanked(self) -> bool:
        # check if beam is blanked or not
        val = self.microscope.optics.is_beam_blanked
        logging.info(f"Checking: Is beam blanked?-- current status: {val}")
        return val
    
    def blank_beam(self) -> None:
        # Perform electron beam blanking
        logging.info("Performing beam blanking")
        self.microscope.optics.blank()
        logging.info("DONE -- beam blanking")
        return 

    def unblank_beam(self) -> None:
        # Perform electron beam UNblanking
        logging.info("Performing beam UNblanking")
        self.microscope.optics.unblank()
        logging.info("DONE -- beam UNblanking")
        return 
    
    def query_vacuum_valves(self) -> str:
        # Check status of the column valves
        logging.info("Request: Checking for vacuum valves -- current status")
        val = self.microscope.vacuum.column_valves.state
        logging.info(f"DONE: Checking for vacuum valves -- current status: {val}")
        return val
    
    def open_vacuum_valves(self) -> None:
        #Perform vacuum valves to open
        logging.info("Performing vacuum valves to open")
        self.microscope.vacuum.column_valves.open()
        logging.info("DONE -- Open vacuum valves")
        return 

    def close_vacuum_valves(self) -> None:
        #Perform vacuum valves to open
        logging.info("Performing vacuum valves to close")
        self.microscope.vacuum.column_valves.close()
        logging.info("DONE -- CLose vacuum valves")
        return 

    def query_screen_current(self) -> str:
        # measure screen current 
        logging.info("Measuring screen current")
        val = str(self.microscope.detectors.screen.measure_current() * 1E12) + "pA"
        logging.info(f"DONE -- Querying screen current - value {val}")
        return val
    
    def query_FOV(self) -> Tuple[float, str, str]:
        # query feild of view
        # field of view 80Kx : 1.05522326521168e-06
        # field of view of 225Kx : 3.73888850472222e-07
        # field of view of 450Kx : 1.86944425236111e-07
        # field of view of 1Mx : 8.31111167780556e-08
        # field of view of 2Mx : 4.15555583890278e-08
        # field of view of 4Mx : 2.07777791945139e-08
        # field of view of 8Mx : 1.03888895972569e-08
        # field of view 14.5Mx : 5.79722270188654e-09
        # field of view of 16Mx : 5.19444479862847e-09
        fov_val = self.microscope.optics.scan_field_of_view
        magnification_val_Kx = str((0.08442 /fov_val)/1000)+ "Kx"
        magnification_val_Mx = str((0.08442 /fov_val)/1000000)+ "Mx"

        logging.info(f"DONE querying the field of view - value = {fov_val} which in Magnificatoin is {magnification_val_Kx} in Kx and Mx is {magnification_val_Mx}")
        return fov_val, magnification_val_Kx, magnification_val_Mx
    
    def set_FOV(self, magnification: str) -> None:
        # TODO: Get rid of the dictionary and directly use formula: trivial
        FIELD_OF_VIEW = {
            "80Kx": 1.05522326521168e-06,  # meters
            "225Kx": 3.73888850472222e-07,
            "450Kx": 1.86944425236111e-07,
            "1Mx": 8.31111167780556e-08,
            "2Mx": 4.15555583890278e-08,
            "4Mx": 2.07777791945139e-08,
            "8Mx": 1.03888895972569e-08,
            "14.5Mx": 5.79722270188654e-09,
            "16Mx": 5.19444479862847e-09
                        }
        logging.info(f"Setting Magnification to {magnification}--- in FOV: {FIELD_OF_VIEW[magnification]}")
        self.microscope.optics.scan_field_of_view = FIELD_OF_VIEW[magnification]
        logging.info(f"FOV is now set")
        return


class DMacquisition:
    # acquires HAADF[with scalebar], CBED, EDX
    # also does optistem(C1_A1, C1, B2_A2)
    # later - drift correct on HAADF, defocus series, Segment regions and cbed
    def __init__(self, microscope: Union[Pyro5.api.Proxy, DMtwin], offline: bool = True) -> None:
        """Initialize the microscope and set up acquisition parameters."""
        self.offline = offline
        if not self.offline :
            self.microscope = microscope
            logging.info("initializing Microscope")
            logging.info("activating camera")
            self.microscope.activate_camera()
            array_list, shape, dtype = self.microscope.get_ds(128, 128)#----------- just to start the connection
            im_array = np.array(array_list, dtype=dtype).reshape(shape)# 4.6 seconds
            logging.info("Microscope initialized with dummy DigiScan acquisition")
            
        else:## offline
            logging.info("initializing OFFLINE Microscope")
            self.microscope = microscope
            logging.info("Done: initialized OFFLINE Microscope")
        return None
    
    
    def acquire_eels(self, eels_exposure_seconds : int = 2) -> np.ndarray:
        """Perform EELS acquisition to just check"""
        # TODO: eels_disp_mult --> need to handle
        if not self.offline:
            logging.info(f"Performing EELS")
            self.microscope.acquire_camera(exposure = eels_exposure_seconds)
            # microscope.optics.blank()
            array_list, shape, dtype = self.microscope.get_eels()
            array = np.array(array_list, dtype=dtype).reshape(shape)
            logging.info(f"DONE EELS")
            
        else:# offline
            logging.info(f"Performing EELS")
            # microscope.optics.blank()
            array_list, shape, dtype = self.microscope.get_eels()
            array = np.array(array_list, dtype=dtype).reshape(shape)
            logging.info(f"DONE EELS ")
            
        return array

    def acquire_eels_threading(self, eels_exposure_seconds : int = 2, eels_dispersion : float = 0.3, eels_dispersion_mult : float = 0, eels_cycles = 1) -> np.ndarray:
        """Perform EELS acquisition to across points and get single spectrum"""
        logging.info(f"DONE EELS at dummy points just for checking.")
        return 

# class CEOSacquisition:
#     def __init__(self, microscope: Union[Pyro5.api.Proxy, DMtwin], offline: bool = True) -> None:

# class DTMICacquisition:
#     mic_server = Pyro5.api.Proxy(uri) 
#     mic_server.initialize_microscope("STEM") 
#     mic_server.register_data(dataset_path) 
#     mic_server.get_point_data()

class EDGEfilterAcquisition:
    def __init__(self, microscope):
        """Initialize the microscope and set up acquisition parameters."""
        self.microscope = microscope
        logging.info("Microscope initialized.")

    def acquire_haadf(self):
        """Acquire HAADF image."""
        logging.info("Acquiring HAADF image.")
        # Simulated HAADF acquisition
        return "haadf_image"

    def drift_correct(self, haadf_image):
        """Perform drift correction based on HAADF."""
        logging.info("Performing drift correction.")
        # Simulated drift correction
        return "corrected_haadf"

    def get_edge_coordinates(self, corrected_haadf):
        """Extract edge coordinates from HAADF image."""
        logging.info("Extracting edge coordinates.")
        # Simulated edge detection
        return [(100, 200), (150, 250)]

    def acquire_eels(self, edge_coords):
        """Perform EELS at extracted edge coordinates."""
        logging.info(f"Performing EELS at {edge_coords}.")
        # Simulated EELS acquisition
        return "eels_spectrum"

    def run_pipeline(self):
        """Complete EDGE filter acquisition pipeline."""
        haadf = self.acquire_haadf()
        corrected_haadf = self.drift_correct(haadf)
        edge_coords = self.get_edge_coordinates(corrected_haadf)
        eels_data = self.acquire_eels(edge_coords)
        return eels_data
