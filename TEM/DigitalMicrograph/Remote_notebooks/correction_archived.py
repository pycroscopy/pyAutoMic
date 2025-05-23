# first let's load in the dark current and gain reference
# import pyTEMlib.file_tools as ft
# # from SciFiReaders import  DMReader
# import os
# readout_area = [992, 0, 1056, 2048]

# # load the gain and dark reference
# path = '/Users/utkarshpratiush/Library/CloudStorage/GoogleDrive-upratius@vols.utk.edu/My Drive/random_download/temp_code/ref_images/'
# files = os.listdir(path)

# dark_refs = [f for f in files if 'Dark' in f]
# gain_refs = [f for f in files if 'Gain' in f]

# dark_name = dark_refs[1]
# gain_name = gain_refs[1]

# # below is the dark current for dual eels.  so we split into two halves
# # dark = DMReader(path + dark_name)
# # dark = dark.read()['Channel_000']
# dark = ft.open_file(path + dark_name)['Channel_000']
# dark_zl = dark[:, :64]
# dark_cl = dark[:, 64:]

# # gain is the same size as the readout area
# gain = ft.open_file(path + gain_name)['Channel_000']
# # gain = DMReader(path + gain_name)
# # gain = gain.read()['Channel_000']

# gain = gain[readout_area[1]:readout_area[3], readout_area[0]:readout_area[2]]

# print('dark_zl shape', dark_zl.shape)
# print('dark_cl shape', dark_cl.shape)
# print('gain shape', gain.shape)

# def corrected_spectrum(dark_current, gain_current, uncorrected_spectrum):
#     dark_sum = np.array(np.sum(dark_current, axis=1))
#     dark_sum -= dark_sum.min()
#     gain_sum = np.array(np.sum(gain_current, axis=1))
#     corrected_spectrum = (uncorrected_spectrum - 2*(dark_sum - dark_sum.min())) / gain_sum
#     corrected_spectrum -= corrected_spectrum.min()
    
#     return corrected_spectrum

    