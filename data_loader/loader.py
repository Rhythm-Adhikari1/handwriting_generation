import random
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pickle
from torchvision import transforms
import lmdb
from PIL import Image
import torchvision
import cv2
from einops import rearrange, repeat
import time
import torch.nn.functional as F
from data_loader.devanagari_tokenizer import split_syllables
from data_loader.unicode_generation import generate_syllable_image
import re
import traceback


text_path = {'train':'data/Devanagari Dataset/train_word.txt',
             'test':'data/Devanagari Dataset/test_word.txt'}

generate_type = {'iv_s':['train', 'data/Devanagari Dataset/in_vocab.subset.tro.37'],
                'iv_u':['test', 'data/Devanagari Dataset/in_vocab.subset.tro.37'],
                'oov_s':['train', 'data/Devanagari Dataset/oov.common_words'],
                'oov_u':['test', 'data/Devanagari Dataset/oov.common_words']}

# define the letters and the width of style image
letters = '_Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
style_len = 352

"""prepare the IAM dataset for training"""
class IAMDataset(Dataset):
    def __init__(self, image_path, style_path, laplace_path, type, content_type='unicode_devanagari', max_len=10):
        self.max_len = max_len
        self.style_len = style_len
        self.data_dict = self.load_data(text_path[type])
        self.image_path = os.path.join(image_path, type)
        self.style_path = os.path.join(style_path, type)
        self.laplace_path = os.path.join(laplace_path, type)

        ##self.letters = letters
        ##self.tokens = {"PAD_TOKEN": len(self.letters)}    ## yo kina chainxa herna xa
        ##self.letter2index = {label: n for n, label in enumerate(self.letters)}

        self.indices = list(self.data_dict.keys())
        self.transforms = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
        #self.content_transform = torchvision.transforms.Resize([64, 32], interpolation=Image.NEAREST)
        
        self.con_symbols = self.get_symbols(content_type)
        #self.syllables = list(self.con_symbols.keys())
        #self.syllable2index = {s: i for i, s in enumerate(self.syllables)}

        self.laplace = torch.tensor([[0, 1, 0],[1, -4, 1],[0, 1, 0]], dtype=torch.float
                                    ).to(torch.float32).view(1, 1, 3, 3).contiguous()



    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            train_data = f.readlines()
            train_data = [i.strip().split(' ') for i in train_data]
            full_dict = {}
            idx = 0
            for i in train_data:
                s_id = i[0].split(',')[0]

                image = i[0].split(',')[1]

                if not image.lower().endswith(".png"):
                    image += ".png"

                transcription = i[1]
                transcription = split_syllables(transcription)

                if len(transcription) > self.max_len or len(transcription) == 0 or len(image) == 0 or len(s_id) == 0:
                    continue

                full_dict[idx] = {'image': image, 's_id': s_id, 'label':transcription}
                idx += 1

        return full_dict

    def get_style_ref(self, wr_id):
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        style_index = random.sample(range(len(style_list)), 2) # anchor and positive
        style_images = [cv2.imread(os.path.join(self.style_path, wr_id, style_list[index]), flags=0)
                        for index in style_index]
        laplace_images = [cv2.imread(os.path.join(self.laplace_path, wr_id, style_list[index]), flags=0)
                          for index in style_index]
        
        height = style_images[0].shape[0]
        assert height == style_images[1].shape[0], 'the heights of style images are not consistent'
        max_w = max([style_image.shape[1] for style_image in style_images])
        
        '''style images'''
        style_images = [style_image/255.0 for style_image in style_images]
        new_style_images = np.ones([2, height, max_w], dtype=np.float32)
        new_style_images[0, :, :style_images[0].shape[1]] = style_images[0]
        new_style_images[1, :, :style_images[1].shape[1]] = style_images[1]

        '''laplace images'''
        laplace_images = [laplace_image/255.0 for laplace_image in laplace_images]
        new_laplace_images = np.zeros([2, height, max_w], dtype=np.float32)
        new_laplace_images[0, :, :laplace_images[0].shape[1]] = laplace_images[0]
        new_laplace_images[1, :, :laplace_images[1].shape[1]] = laplace_images[1]
        return new_style_images, new_laplace_images

    def get_symbols(self, input_type):
        with open(f"data/Devanagari Dataset/{input_type}.pickle", "rb") as f:
            data = pickle.load(f)

        
        #symbols = {sym['label']: sym['mat'].astype(np.float32) for sym in symbols}
        

        symbols = {}
        for sym in data:
            label = ''.join(chr(i) for i in sym['idx'])
            symbols[label] = torch.from_numpy(sym['mat'].astype(np.float32))

        contents = list(symbols.values())
        self.syllables = list(symbols.keys())
        self.syllable2index = {s: i for i, s in enumerate(self.syllables)}
        
        # for char in self.letters:
        #     symbol = torch.from_numpy(symbols[ord(char)]).float()
        #     contents.append(symbol)

        #contents.append(torch.zeros_like(contents[0])) # blank image as PAD_TOKEN # why necessary need to look ; might be essential
        contents = torch.stack(contents)
        return contents
       
    def __len__(self):
        return len(self.indices)
    
    def pad_to_multiple(self, img, base=32):
        _, h, w = img.shape
        pad_h = (base - h % base) % base
        pad_w = (base - w % base) % base
        img = F.pad(img, (0, pad_w, 0, pad_h), value=1.0)
        return img


    ### Borrowed from GANwriting ###
    def label_padding(self, labels, max_len):
        ll = [self.syllable2index[i] for i in labels]
        num = max_len - len(ll)
        if not num == 0:
            ll.extend([self.tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
        return ll

    def __getitem__(self, idx):
        image_name = self.data_dict[self.indices[idx]]['image']
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['s_id']
        
        transcr = label.copy()
        img_path = os.path.join(self.image_path, wr_id, image_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)

        style_ref, laplace_ref = self.get_style_ref(wr_id)
        style_ref = torch.from_numpy(style_ref).to(torch.float32) # [2, h , w] achor and positive
        laplace_ref = torch.from_numpy(laplace_ref).to(torch.float32) # [2, h , w] achor and positive
        image = self.pad_to_multiple(image)

        return {'img':image,
                'content':label, 
                'style':style_ref,
                "laplace":laplace_ref,
                'wid':int(wr_id),
                'transcr':transcr,
                'image_name':image_name}


    def collate_fn_(self, batch):
        width = [item['img'].shape[2] for item in batch]
        c_width = [len(item['content']) for item in batch]
        s_width = [item['style'].shape[2] for item in batch]

        transcr = [item['transcr'] for item in batch]
        target_lengths = torch.IntTensor([len(t) for t in transcr])
        image_name = [item['image_name'] for item in batch]

        # Force fixed style and image width to 352
        if max(s_width) < self.style_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.style_len

        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)], dtype=torch.float32)
        content_ref = torch.zeros([len(batch), max(c_width), 16 , 16], dtype=torch.float32)
        
        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width], dtype=torch.float32)
        laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], batch[0]['laplace'].shape[1], max_s_width], dtype=torch.float32)
        target = torch.zeros([len(batch), max(target_lengths)], dtype=torch.int32)

        for idx, item in enumerate(batch):
            # --- pad/truncate style & laplace ---
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print('img', item['img'].shape)

            # --- content symbols ---
            content_tensor = []
            # if not item['content']:
            #     item['content'].append('|')
            try:
                for syl in item['content']:
                    if syl not in self.syllable2index.keys():
                        img = generate_syllable_image(syl)
                        img_tensor = torch.from_numpy(np.array(img, dtype=np.float32)) / 255.0
                        self.con_symbols = torch.cat([self.con_symbols, img_tensor.unsqueeze(0)], dim=0)
                        new_index = len(self.syllables)
                        self.syllables.append(syl)
                        self.syllable2index[syl] = new_index
                    content_tensor.append(self.con_symbols[self.syllable2index[syl]])

                content_tensor = torch.stack(content_tensor)
                content_ref[idx, :len(content_tensor)] = content_tensor

            except Exception as e:
                print(f"⚠️ content error for {item['image_name']}: {e}")

            # --- transcription labels ---
            target[idx, :len(transcr[idx])] = torch.Tensor(
                [self.syllable2index[t] for t in transcr[idx]]
            )

            

            try:
                if max_s_width < self.style_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace']
                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.style_len]
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace'][:, :, :self.style_len]
            except:
                print('style', item['style'].shape)


        wid = torch.tensor([item['wid'] for item in batch])
        content_ref = 1.0 - content_ref  # invert image

        return {
            'img': imgs,
            'style': style_ref,
            'content': content_ref,
            'wid': wid,
            'laplace': laplace_ref,
            'target': target,
            'target_lengths': target_lengths,
            'image_name': image_name
        }


"""random sampling of style images during inference"""
class Random_StyleIAMDataset(IAMDataset):
    def __init__(self, style_path, lapalce_path, ref_num) -> None:
        self.style_path = style_path
        self.laplace_path = lapalce_path
        self.author_id = os.listdir(os.path.join(self.style_path))
        self.style_len = style_len
        self.ref_num = ref_num
    
    def __len__(self):
        return self.ref_num
    
    def get_style_ref(self, wr_id): # Choose the style image whose length exceeds 32 pixels
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        random.shuffle(style_list)
        for index in range(len(style_list)):
            style_ref = style_list[index]

            style_image = cv2.imread(os.path.join(self.style_path, wr_id, style_ref), flags=0)
            laplace_image = cv2.imread(os.path.join(self.laplace_path, wr_id, style_ref), flags=0)
            if style_image.shape[1] > 128:
                break
            else:
                continue
        style_image = style_image/255.0
        laplace_image = laplace_image/255.0
        return style_image, laplace_image

    def __getitem__(self, _):
        batch = []
        for idx in self.author_id:
            style_ref, laplace_ref = self.get_style_ref(idx)
            style_ref = torch.from_numpy(style_ref).unsqueeze(0)
            style_ref = style_ref.to(torch.float32)
            laplace_ref = torch.from_numpy(laplace_ref).unsqueeze(0)
            laplace_ref = laplace_ref.to(torch.float32)
            wid = idx
            batch.append({'style':style_ref, 'laplace':laplace_ref, 'wid':wid})
        
        s_width = [item['style'].shape[2] for item in batch]
        if max(s_width) < self.style_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.style_len
        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width], dtype=torch.float32)
        laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], batch[0]['laplace'].shape[1], max_s_width], dtype=torch.float32)
        wid_list = []
        for idx, item in enumerate(batch):
            try:
                if max_s_width < self.style_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace']
                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.style_len]
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace'][:, :, :self.style_len]
                wid_list.append(item['wid'])
            except:
                print('style', item['style'].shape)
        
        return {'style':style_ref, 'laplace':laplace_ref,'wid':wid_list}

"""prepare the content image during inference"""    
class ContentData(IAMDataset):
    def __init__(self, content_type='unifont') -> None:
        self.letters = letters
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.con_symbols = self.get_symbols(content_type)
       
    def get_content(self, label):
        word_arch = [self.letter2index[i] for i in label]
        content_ref = self.con_symbols[word_arch]
        content_ref = 1.0 - content_ref
        return content_ref.unsqueeze(0)