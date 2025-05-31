import os
from glob import glob
from collections import defaultdict
from PIL import Image

# inspect the data in ./data/ load five of each subfolder

data_dir = "./data/train"
samples_per_class = 5

data_samples = defaultdict(list)

for class_dir in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_dir)
    if os.path.isdir(class_path):
        files = glob(os.path.join(class_path, "*"))
        data_samples[class_dir] = files[:samples_per_class]

        # fig, axs = plt.subplots(
        #     len(data_samples),
        #     samples_per_class,
        #     figsize=(samples_per_class * 2, len(data_samples) * 2),
        # )

        # for row_idx, (class_name, file_list) in enumerate(data_samples.items()):
        #     for col_idx, file_path in enumerate(file_list):
        #         img = Image.open(file_path)
        #         ax = axs[row_idx, col_idx] if len(data_samples) > 1 else axs[col_idx]
        #         ax.imshow(img)
        #         ax.axis("off")
        #         if col_idx == 0:
        #             ax.set_title(class_name)
        # plt.tight_layout()
        # plt.show()

        for file_path in data_samples[class_dir]:
            with Image.open(file_path) as img:
                print(
                    f"Class: {class_dir}, "
                    f"File: {os.path.basename(file_path)}, "
                    f"Size: {img.size}, "
                    f"Mode: {img.mode}"
                )
        print()
