import blobfile as bf
from PIL import Image

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

dataset_name = 'beihong'
author_name = " by Beihong Xu"

files = _list_image_files_recursively("./{}/original_data/".format(dataset_name))

template = '''{"file_name": "TEMP_ID.png", "text": "text_temp"}\n'''
print(template)

line_list = []

for i, file_path in enumerate(files):
    file_name = file_path.split('/')[-1]
    caption = file_name.strip('.jpg').replace('-', ' ')
    caption = caption[0].upper()+caption[1:].lower()+author_name

    with bf.BlobFile(file_path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()

        min_size = min(pil_image.size[0], pil_image.size[1], )
        if min_size < 550:
            newsize = (int(pil_image.size[0] /min_size * 600), int(pil_image.size[1] /min_size * 600), )
            pil_image = pil_image.resize(newsize)
        print(pil_image.size)

        save_path = './{}/dataset/train/{}.png'.format(dataset_name, '{:0>4}'.format(i+1))
        pil_image.convert("RGB").save(save_path)
        line = template.replace('TEMP_ID', '{:0>4}'.format(i+1)).replace('text_temp', caption)
        line_list.append(line)

with open("./{}/dataset/train/metadata.jsonl".format(dataset_name),'w') as f:
    f.writelines(line_list)
