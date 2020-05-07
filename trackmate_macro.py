from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.action import ExportAllSpotsStatsAction
from fiji.plugin.trackmate.detection import LogDetectorFactory
import fiji.plugin.trackmate.detection.DogDetectorFactory as DogDetectorFactory
from fiji.plugin.trackmate.tracking.kdtree import NearestNeighborTrackerFactory
from fiji.plugin.trackmate.tracking import LAPUtils
from ij import IJ
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter
import sys
import fiji.plugin.trackmate.features.track.TrackDurationAnalyzer as TrackDurationAnalyzer
from fiji.plugin.trackmate.tracking.sparselap import SparseLAPTrackerFactory
from fiji.plugin.trackmate.tracking import LAPUtils
import os
import glob
from ij import IJ, ImagePlus, ImageStack
import fiji.plugin.trackmate.action as action
from fiji.plugin.trackmate.action import ExportStatsToIJAction
from fiji.plugin.trackmate.action import ExportAllSpotsStatsAction
import java.util.ArrayList as ArrayList
import csv
from fiji.plugin.trackmate import Spot
import copy

os.chdir("D:\\SinglemoleculeRawData_others\\Myr\\")
filenames = glob.glob("*")

def my_makedirs(path):
	if not os.path.isdir(path):
		os.makedirs(path)
#for filename in filenames[17:-1]:
for filename in filenames[1:]:
	path = os.path.join("C:\\Users\\matsuno\\Desktop\\Jupyternotebook\\LAPTracker\\GapFalse_DoG\\{}\\Spots\\".format(filename))
	my_makedirs(path)
#	os.makedirs(os.path.join("C:\\Users\\matsuno\\Desktop\\Jupyternotebook\\LAPTracker\\{}\\Spots".format(filename)))
	os.chdir("D:\\SinglemoleculeRawData_others\\Myr\\{}\\Crop".format(filename))
	files = glob.glob("*.tif")
	print(filename)
	for file in files[:]:
		print(file)
		imp = IJ.openImage(os.path.join("D:\\SinglemoleculeRawData_others\\Myr\\{}\\Crop".format(filename), file))
		IJ.run(imp, "Properties...", "channels=1 slices=1 frames=900 unit=um pixel_width=0.067 pixel_height=0.067 voxel_depth=1 global");
		#----------------------------
		# Create the model object now
		#----------------------------
		# Some of the parameters we configure below need to have
		# a reference to the model at creation. So we create an
		# empty model now.
		model = Model()
		    
		# Send all messages to ImageJ log window.
		model.setLogger(Logger.IJ_LOGGER)
		
		#------------------------
		# Prepare settings object
		#------------------------
		settings = Settings()
		settings.setFrom(imp)
		
		# Configure detector - We use the Strings for the keys
		settings.detectorFactory = LogDetectorFactory()
		settings.detectorSettings = { 
		    'DO_SUBPIXEL_LOCALIZATION' : True,
		    'RADIUS' : 0.125,
		    'TARGET_CHANNEL' : 1,
		    'THRESHOLD' : 150.,
		    'DO_MEDIAN_FILTERING' :True,
		}
		
		# Configure tracker - We want to allow merges and fusions
		settings.trackerFactory = SparseLAPTrackerFactory()
		settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap() # almost good enough
		settings.trackerSettings['LINKING_MAX_DISTANCE'] = 0.67
		settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = False
		settings.trackerSettings['ALLOW_TRACK_MERGING'] = False
		settings.trackerSettings['ALLOW_GAP_CLOSING'] = False
#		settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 0.1
#		settings.trackerSettings['MAX_FRAME_GAP']= 1
		settings.addTrackAnalyzer(TrackDurationAnalyzer())
		
		filter2= FeatureFilter('TRACK_DURATION', 5, True)
		settings.addTrackFilter(filter2)
		
		#-------------------
		# Instantiate plugin
		#-------------------
		    
		trackmate = TrackMate(model, settings)
		       
		#--------
		# Process
		#--------
		    
		ok = trackmate.checkInput()
		if not ok:
		    sys.exit(str(trackmate.getErrorMessage()))
		    
		ok = trackmate.process()
		if not ok:
		    sys.exit(str(trackmate.getErrorMessage()))
		
		#----------------
		# Display results
		#----------------
#		model.getLogger().log('Found ' + str(model.getTrackModel().nTracks(True)) + ' tracks.')
#		selectionModel = SelectionModel(model)
#		displayer =  HyperStackDisplayer(model, selectionModel, imp)
#		displayer.render()
#		displayer.refresh()
		
		# The feature model, that stores edge and track features.
		fm = model.getFeatureModel()
				
		id_num = 0
		with open('C:\\Users\\matsuno\\Desktop\\Jupyternotebook\\LAPTracker\\GapFalse_DoG\\{}\\Spots\\testdog'.format(filename) + file[:-4]  +'_spots.csv', 'wb') as f:
		    writer = csv.writer(f)
		    writer.writerow(["TrackID", "FRAME", "POSITION_X", "POSITION_Y", "QUALITY"])
		    for id in model.getTrackModel().trackIDs(True):
			    tracks = model.getTrackModel().trackSpots(id)
			    tracks = ArrayList(tracks);
			    tracks2 = []
			    for spot in tracks:
			        tracks2.append([
		                id_num, spot.getFeature('FRAME'), spot.getFeature('POSITION_X'), 
		                spot.getFeature('POSITION_Y'), spot.getFeature('QUALITY')
		            ])
			    track_sorted = sorted(tracks2, key=lambda x: x[1], reverse=False)
#			    start_frame = copy.deepcopy(track_sorted)[0][1]
#			    for i in range(len(track_sorted)):
#			        track_sorted[i][1] -= start_frame
			    for j in range(len(track_sorted)):
			        writer.writerow([track_sorted[j][0], track_sorted[j][1], track_sorted[j][2], track_sorted[j][3], track_sorted[j][4]])
			    id_num += 1
	        
		ExportAllSpotsStatsAction().execute(trackmate)
		directory = "C:\\Users\\matsuno\\Desktop\\Jupyternotebook\\LAPTracker\\GapFalse_DoG\\{}\\testdog".format(filename)
		IJ.saveAs("text", directory + file[:-4] + ".csv")
		IJ.run("Close");
		imp.close();

print("NN is finished")

#for filename in filenames[:1]:
#	print(filename)
#	path = os.path.join("C:\\Users\\matsuno\\Desktop\\Jupyternotebook\\LAPTracker\\GapAllow1Frame0.1\\RawData\\{}\\Spots\\".format(filename))
#	my_makedirs(path)
##	os.makedirs(os.path.join("C:\\Users\\matsuno\\Desktop\\Jupyternotebook\\LAPTracker\\{}\\Spots".format(filename)))
#	os.chdir("D:\\SinglemoleculeRawData\\{}\\Crop".format(filename))
#	files = glob.glob("*.tif")
#	for file in files[:]:
#		imp = IJ.openImage(os.path.join("D:\\SinglemoleculeRawData\\{}\\Crop".format(filename), file))
#		IJ.run(imp, "Properties...", "channels=1 slices=1 frames=900 unit=um pixel_width=0.067 pixel_height=0.067 voxel_depth=1 global");
#		
#		#----------------------------
#		# Create the model object now
#		#----------------------------
#		# Some of the parameters we configure below need to have
#		# a reference to the model at creation. So we create an
#		# empty model now.
#		model = Model()
#		    
#		# Send all messages to ImageJ log window.
#		model.setLogger(Logger.IJ_LOGGER)
#		
#		#------------------------
#		# Prepare settings object
#		#------------------------
#		settings = Settings()
#		settings.setFrom(imp)
#		
#		# Configure detector - We use the Strings for the keys
#		settings.detectorFactory = LogDetectorFactory()
#		settings.detectorSettings = { 
#		    'DO_SUBPIXEL_LOCALIZATION' : True,
#		    'RADIUS' : 0.25,
#		    'TARGET_CHANNEL' : 1,
#		    'THRESHOLD' : 35.,
#		    'DO_MEDIAN_FILTERING' : False,
#		}
#		
#		# Configure tracker - We want to allow merges and fusions
#		settings.trackerFactory = SparseLAPTrackerFactory()
#		settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap() # almost good enough
#		settings.trackerSettings['LINKING_MAX_DISTANCE'] = 0.67
#		settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = False
#		settings.trackerSettings['ALLOW_TRACK_MERGING'] = False
#		settings.trackerSettings['ALLOW_GAP_CLOSING'] = True
#		settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 0.1
#		settings.trackerSettings['MAX_FRAME_GAP']= 1
#		settings.addTrackAnalyzer(TrackDurationAnalyzer())
#		
#		filter2= FeatureFilter('TRACK_DURATION', 5, True)
#		settings.addTrackFilter(filter2)
#		
#		#-------------------
#		# Instantiate plugin
#		#-------------------
#		trackmate = TrackMate(model, settings)
#		
#		#--------
#		# Process
#		#--------
#		ok = trackmate.checkInput()
#		if not ok:
#		    sys.exit(str(trackmate.getErrorMessage()))
#		ok = trackmate.process()
#		if not ok:
#		    sys.exit(str(trackmate.getErrorMessage()))
#		
#		# The feature model, that stores edge and track features.
#		fm = model.getFeatureModel()
#		
#		from ij import IJ, ImagePlus, ImageStack
#		import fiji.plugin.trackmate.action as action
#		from fiji.plugin.trackmate.action import ExportStatsToIJAction
#		from fiji.plugin.trackmate.action import ExportAllSpotsStatsAction
#		import java.util.ArrayList as ArrayList
#		import csv
#		from fiji.plugin.trackmate import Spot
#		import copy
#		
#		id_num = 0
#		with open('C:\\Users\\matsuno\\Desktop\\Jupyternotebook\\LAPTracker\\GapAllow1Frame0.1\\RawData\\{}\\Spots\\NoMedian'.format(filename) + file[:-4]  +'_spots.csv', 'wb') as f:
#		    writer = csv.writer(f)
#		    writer.writerow(["TrackID", "FRAME", "POSITION_X", "POSITION_Y", "QUALITY"])
#		    for id in model.getTrackModel().trackIDs(True):
#			    tracks = model.getTrackModel().trackSpots(id)
#			    tracks = ArrayList(tracks);
#			    tracks2 = []
#			    for spot in tracks:
#			        tracks2.append([
#		                id_num, spot.getFeature('FRAME'), spot.getFeature('POSITION_X'), 
#		                spot.getFeature('POSITION_Y'), spot.getFeature('QUALITY')
#		            ])
#			    track_sorted = sorted(tracks2, key=lambda x: x[1], reverse=False)
##			    start_frame = copy.deepcopy(track_sorted)[0][1]
##			    for i in range(len(track_sorted)):
##			        track_sorted[i][1] -= start_frame
#			    for j in range(len(track_sorted)):
#			        writer.writerow([track_sorted[j][0], track_sorted[j][1], track_sorted[j][2], track_sorted[j][3], track_sorted[j][4]])
#			    id_num += 1
#	        
#		ExportAllSpotsStatsAction().execute(trackmate)
#		directory = "C:\\Users\\matsuno\\Desktop\\Jupyternotebook\\LAPTracker\\GapAllow1Frame0.1\\RawData\\{}\\NoMedian".format(filename)
#		IJ.saveAs("text", directory + file[:-4] + ".csv")
#		IJ.run("Close");
#		imp.close();

print("finish")