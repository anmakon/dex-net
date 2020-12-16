import numpy as np

RATIO_POS = 1.0
METRIC_TRESH = 0.002
DATA_DIR = '/data'


class CreateValidationPointers:
    def __init__(self):
        self.path = DATA_DIR + "/dexnet_2_tensor/tensors/"
        self.output_file = DATA_DIR + "/reprojections/generated_val_indices.txt"
        self.indices_file = DATA_DIR + "/dexnet_2_tensor/splits/image_wise/val_indices.npz"
        self.tensor_file = "./data/excellent_predictions.txt"
        self.random = False  # False --> take pointers from tensor_file; True --> generate random pointers
        self.tensor = 0
        self.array = 0
        self.main()

    def _get_pose_number(self, obj_label):
        # This is to get the pose number for the chosen datapoint. As the pose labels are incrementing over the
        # whole dataset, subtract the current pose label from the first pose label of the object
        # This ain't working. Check it!

        old_tensor = self.tensor
        obj_labels = np.load(self.path + "object_labels_" + self._filepointer() + ".npz")['arr_0']
        pose_label = np.load(self.path + "pose_labels_" + self._filepointer() + ".npz")['arr_0'][self.array]

        # Iterate until there is an object label smaller than the datapoint object label in the tensor
        while not np.any(obj_labels < obj_label):
            self.tensor -= 1
            obj_labels = np.load(self.path + "object_labels_" + self._filepointer() + ".npz")['arr_0']

        # Increase the tensor if there are no object labels fitting the datapoint object label
        if not np.any(obj_labels == obj_label):
            self.tensor += 1
            obj_labels = np.load(self.path + "object_labels_" + self._filepointer() + ".npz")['arr_0']

        # Take the first datapoint where the object label matches
        match = np.where(obj_labels == obj_label)[0]
        position = match[0]

        # Load the pose label
        first_pose_label = np.load(self.path + "pose_labels_" + self._filepointer() + ".npz")['arr_0'][position]

        # Compute the pose number
        pos_num = pose_label - first_pose_label
        self.tensor = old_tensor
        return pos_num

    def _get_grasp_number(self, image_label):
        # This is to get the grasp number. Each grasp is presented in each image. To get the grasp number,
        # count the position of the datapoint in the datapoints with the same image label.

        old_tensor = self.tensor
        image_labels = np.load(self.path + "image_labels_" + self._filepointer() + ".npz")['arr_0']
        reduced = False

        # Iterate if you don't find any image label in the tensor that is smaller than the datapoint image label
        i = 0
        while not np.any(image_labels < image_label):
            self.tensor -= 1
            image_labels = np.load(self.path + "image_labels_" + self._filepointer() + ".npz")['arr_0']
            reduced = True
            i += 1
            if i >= 2:
                raise NotImplementedError("No smaller image label than %d in tensor %d" %(image_label, self.tensor))

        # Increase the tensor if there are no image labels fitting the datapoint image label
        if not np.any(image_labels == image_label):
            self.tensor += 1
            image_labels = np.load(self.path + "image_labels_" + self._filepointer() + ".npz")['arr_0']

        match = np.where(image_label == image_labels)[0]
        # Get the grasp number
        if reduced:
            if type(match) == int:
                grasp_num = self.array + len(image_labels) - match
            else:
                grasp_num = self.array + len(image_labels) - match[0]
        else:
            grasp_num = self.array - match[0]

        self.tensor = old_tensor
        return grasp_num

    def _filepointer(self):
        return "{0:05d}".format(self.tensor)

    def main(self):
        val_ind = []
        if self.random:
            validation_indices = np.load(self.indices_file)['arr_0']
            n = int(input("How many validation indices should be taken? "))
            pos = 0

            while len(val_ind) < n:
                index = np.random.choice(validation_indices)
                self.tensor = index // 1000
                self.array = index % 1000
                metric = np.load(self.path + "robust_ferrari_canny_" + self._filepointer() + ".npz")['arr_0'][self.array]
                if (pos <= int(n * RATIO_POS)) and (metric > METRIC_TRESH):
                    pos += 1
                    val_ind.append(index)
                elif (pos > int(n * RATIO_POS)) and (metric <= METRIC_TRESH):
                    val_ind.append(index)
        else:
            tensorfile = open(self.tensor_file, 'r')
            for line in tensorfile:
                if not 'Tensor' in line:
                    self.tensor = int(line.split(',')[0])
                    self.array = int(line.split(',')[1])
                    val_ind.append([self.tensor, self.array])

        f = open(self.output_file, "w")
        f.write("Tensor, Array, Object_label, Pose_num, Grasp_num, Prev_object_label, Robustness\n")

        for index in val_ind:
            if type(index) == list:
                self.tensor = index[0]
                self.array = index[1]
            else:
                self.tensor = index // 1000
                self.array = index % 1000

            obj_label = np.load(self.path + "object_labels_" + self._filepointer() + ".npz")['arr_0'][self.array]
            image_label = np.load(self.path + "image_labels_" + self._filepointer() + ".npz")['arr_0'][self.array]
            pose_num = self._get_pose_number(obj_label)
            grasp_num = self._get_grasp_number(image_label)
            metric = np.load(self.path + "robust_ferrari_canny_" + self._filepointer() + ".npz")['arr_0'][self.array]

            f.write("%d,%d,%d,%d,%d,%d\n" % (self.tensor, self.array, obj_label, pose_num, grasp_num, metric))
        f.close()


if __name__ == "__main__":
    CreateValidationPointers()
