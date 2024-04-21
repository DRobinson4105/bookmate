import os

def filter_directory(dataset_path, class_number):
    images_path = os.path.join(dataset_path, 'images')
    labels_path = os.path.join(dataset_path, 'labels')

    for filename in os.listdir(labels_path):
        label_path = os.path.join(labels_path, filename)
        image_path = os.path.join(images_path, filename[:-3] + "jpg")

        with open(label_path, 'r') as file:
            lines = file.readlines()
            result = []
            for line in lines:
                parts = line.split()
                if int(parts[0]) == class_number:
                    result.append(" ".join(parts))
            
        if len(result) == 0:
            os.remove(label_path)
            os.remove(image_path)
        else:
            with open(label_path, 'w') as file:
                file.write("\n".join(result))

if __name__ == '__main__':
    filter_directory('./datasets/barcodes/test', 0)
    print('Test directory filtered')
    filter_directory('./datasets/barcodes/train', 0)
    print('Train directory filtered')
    filter_directory('./datasets/barcodes/valid', 0)
    print('Valid directory filtered')