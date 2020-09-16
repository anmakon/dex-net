import numpy as np

# Script to sample validation grasping points from the DexNet 2.0 dataset.
# Amount of samples given through input during script execution
# Ratio of positive and negative ground truth grasps can be varied by changing
# the global variable RATIO_POS.

RATIO_POS = 1
METRIC_TRESH = 0.002

class CreateValidationPointers():
	def __init__(self):
		dataset_path = "/media/psf/Home/Documents/Grasping_Research/Datasets/dexnet_2_tensor/"
		self.data_path = dataset_path+"tensors/"
		self.output_file = "data/generated_val_indices.txt"
		self.indices_file = dataset_path+"splits/image_wise/val_indices.npz"
		self.main()

	def _get_pose_number(self,obj_label):
		""" During dataset generation, the pose is incremented for every new stable pose.
		To get the identifier of the stable pose of a specific validation grasping point,
		the pose number has to be subtracted by the pose number of the last stable pose in the previous
		object.
		Returns the identifier of the stable pose within the given object.
		"""
		old_tensor = self.tensor
		obj_labels = np.load(self.data_path+"object_labels_"+self._filepointer()+".npz")['arr_0']
		pose_label = np.load(self.data_path+"pose_labels_"+self._filepointer()+".npz")['arr_0'][self.array]
		while not np.any(obj_labels < obj_label):
			self.tensor -= 1
			obj_labels = np.load(self.data_path+"object_labels_"+self._filepointer()+".npz")['arr_0']
		match = np.where(obj_label-1 == obj_labels)[0]
		try:
			position = match[-1]
		except IndexError:
			position = match
		prev_obj_label = np.load(self.data_path+"object_labels_"+self._filepointer()+".npz")['arr_0'][position]
		if position == 999:
			print("Special case")
			self.tensor += 1
			first_pose_label = np.load(self.data_path+"pose_labels_"+self._filepointer()+".npz")['arr_0'][0]
		else:
			first_pose_label = np.load(self.data_path+"pose_labels_"+self._filepointer()+".npz")['arr_0'][position+1]
		pos_num = pose_label-first_pose_label
		self.tensor = old_tensor
		return prev_obj_label, pos_num
	
	def _get_grasp_number(self,image_label):
		""" Get identifier of grasp position. Grasp position is being iterated for every new grasp during dataset
		generation. To get the grasp position, the image label of the validation grasping point has to be subtracted
		by the last image label of the previous stable pose in the dataset.
		Returns the identifier of the grasp position within the stable pose.
		"""
		old_tensor = self.tensor
		image_labels = np.load(self.data_path+"image_labels_"+self._filepointer()+".npz")['arr_0']
		if np.any(image_labels < image_label):
			match = np.where(image_label == image_labels)[0]
			grasp_num = self.array-match[0]
		else:
			self.tensor -= 1
			image_labels = np.load(self.data_path+"image_labels_"+self._filepointer()+".npz")['arr_0']
			match = np.where(image_label == image_labels)[0]
			grasp_num = self.array+len(image_labels)-match[0]	
		self.tensor = old_tensor
		return grasp_num
			
			
	def _filepointer(self):
		return ("{0:05d}").format(self.tensor)

	def main(self):
		validation_indices = np.load(self.indices_file)['arr_0']
		n = int(input("How many validation indices should be taken? "))

		val_ind = []
		pos = 0

		while len(val_ind)<n:
			# Sampling random indices from the validation datapoints
			index = np.random.choice(validation_indices)
			self.tensor = index // 1000
			self.array = index % 1000
			metric = np.load(self.data_path+"robust_ferrari_canny_"+self._filepointer()+".npz")['arr_0'][self.array]
			# Checking if the point has the right ground truth for the desired ratio
			if (pos < int(n * RATIO_POS)) and (metric > METRIC_TRESH):
				print( metric)
				pos += 1
				val_ind.append(index)
			elif (pos >= int(n * RATIO_POS)) and (metric <= METRIC_TRESH):
				val_ind.append(index)

		f = open(self.output_file,"w")
		f.write("Tensor, Array, Object_label, Pose_num, Grasp_num, Prev_object_label\n")

		for index in val_ind:
			# Iterate through indices
			self.tensor = index // 1000
			self.array = index % 1000

			obj_label = np.load(self.data_path+"object_labels_"+self._filepointer()+".npz")['arr_0'][self.array]
			image_label = np.load(self.data_path+"image_labels_"+self._filepointer()+".npz")['arr_0'][self.array]
			prev_obj_label,pose_num = self._get_pose_number(obj_label)
			grasp_num = self._get_grasp_number(image_label)

			# Write data into file
			f.write("%d,%d,%d,%d,%d,%d\n"%(self.tensor,self.array,obj_label,pose_num,grasp_num,prev_obj_label))
		f.close()

if __name__ == "__main__":
	CreateValidationPointers()
