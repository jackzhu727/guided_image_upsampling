import Metashape
import os
print(Metashape.app.enumGPUDevices())
Metashape.app.gpu_mask = 1
if not Metashape.app.activated:
        raise('No license')
doc = Metashape.Document()
todo_list = range(20, 30)
pic_dir = ''

for num in todo_list:
	chunk = doc.addChunk()
	chunk.label = "pic_{}".format(num)
	path_photos = pic_dir + "pic_{}".format(num)
	print(path_photos)

	image_list = os.listdir(path_photos)
	chunk.addPhotos([path_photos + '/' + x for x in image_list if x.split('.')[-1].lower() == 'jpg'])

	for carmera in chunk.cameras:
		if "GOPR" in carmera.label:
			carmera.sensor.type = Metashape.Sensor.Type.Fisheye


	for frame in chunk.frames:
	    frame.matchPhotos(accuracy=Metashape.HighestAccuracy, keypoint_limit=40000, tiepoint_limit=40000) # HighestAccuracy

	chunk.alignCameras()

	chunk.buildDepthMaps(quality=Metashape.UltraQuality, filter=Metashape.MildFiltering)
	chunk.buildDenseCloud()

	chunk.buildModel(interpolation=Metashape.DisabledInterpolation, source=Metashape.DenseCloudData, face_count=Metashape.HighFaceCount, vertex_colors=True, keep_depth=True)


	task = Metashape.Tasks.ExportDepth(cameras=chunk.cameras, export_depth=True, path_depth=path_photos, export_normals=False, export_diffuse=True, path_diffuse=path_photos)
	task.apply(chunk)
