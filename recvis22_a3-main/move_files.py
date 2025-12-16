# import os 
# archive = [f for f in os.listdir('imagenet1K/train/archive') if f != '.DS_Store']
# archive1 = [f for f in os.listdir('imagenet1K/train/archive (1)') if f != '.DS_Store']
# archive2 = [f for f in os.listdir('imagenet1K/train/archive (2)') if f != '.DS_Store']
# archive3 = [f for f in os.listdir('imagenet1K/train/archive (3)') if f != '.DS_Store']




# for folder in archive:
#     files = os.listdir(f'imagenet1K/train/archive/{folder}')
#     for file in files:
#         os.makedirs(f'imagenet1K/train/{folder}', exist_ok=True)
#         os.rename(f'imagenet1K/train/archive/{folder}/{file}', f'imagenet1K/train/{folder}/{file}')

# import os 
# count = 0
# for i in range(1000):
#   count+= len(os.listdir(f'imagenet1K/train/{i:05d}'))

# print(count)