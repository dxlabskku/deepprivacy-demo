# IMPORT PACKAGES
from flask import Flask, send_from_directory, jsonify, request
import logging
import cv2
import glob
import torch
import fractions
import numpy as np
import sys
import time
from tqdm import tqdm, trange
import datetime
import pickle
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.reverse2original import reverse2wholeimage
from util.add_watermark import watermark_image
import os
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
import gc
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from celery import Celery, Task
import argparse
from random import choice
import warnings

# HELPER FUNCTION
def lcm(a, b):
    return abs(a * b) / fractions.gcd(a, b) if a and b else 0

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def _toarctensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def config():
    opt = TestOptions().parse()
    crop_size = opt.crop_size
    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    model = create_model(opt)
    model.eval()
    mse = torch.nn.MSELoss().to(device)
    spNorm = SpecificNorm()

    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.1, det_size=(320, 320), mode=mode)
    return app, opt, crop_size, model, mse, spNorm, logoclass

def compute_embedding_distance(face_emb, model, mse, specific_person_id_nonorm, index,target_id_list):
    align_crop_tensor_arcnorm = face_emb[target_id_list[index]].to(device)
    align_crop_tensor_arcnorm_downsample = F.interpolate(align_crop_tensor_arcnorm, size=(112, 112))
    align_crop_id_nonorm = model.netArc(align_crop_tensor_arcnorm_downsample)
    return mse(align_crop_id_nonorm, specific_person_id_nonorm).detach().cpu().numpy()

def process_batch(crops):
    results = [process_crop(crop) for crop in crops]
    return results

def process_crop(spNorm, model, mse, b_align_crop, specific_person_id_nonorm):
    # print(b_align_crop.shape)
    b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB))[None, ...].to(device)
    b_align_crop_tenor_arcnorm = spNorm(b_align_crop_tenor)
    b_align_crop_tenor_arcnorm_downsample = F.interpolate(b_align_crop_tenor_arcnorm, size=(112, 112))
    b_align_crop_id_nonorm = model.netArc(b_align_crop_tenor_arcnorm_downsample.to(device))
    mse_value = mse(b_align_crop_id_nonorm, specific_person_id_nonorm).detach().cpu().numpy()
    return mse_value, b_align_crop_tenor

def get_embedding_specific_person(image, simswap, crop_size, model):
    transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    specific_person_whole = image
    try:
        print(specific_person_whole.shape)
    except AttributeError:
        return 
    
    specific_person_align_crop, _ = simswap.get(specific_person_whole, crop_size)
    specific_person_align_crop_pil = Image.fromarray(cv2.cvtColor(specific_person_align_crop[0], cv2.COLOR_BGR2RGB))
    specific_person = transformer_Arcface(specific_person_align_crop_pil)
    specific_person = specific_person.view(-1, specific_person.shape[0], specific_person.shape[1], specific_person.shape[2])
    specific_person = specific_person.to(device)
    specific_person_downsample = F.interpolate(specific_person, size=(112, 112))
    specific_person_id_nonorm = model.netArc(specific_person_downsample)
    specific_person_id_norm = F.normalize(specific_person_id_nonorm, p=2, dim=1)
    return specific_person_align_crop, specific_person_id_nonorm

def target_hu_inwhole(img_pic_whole, simswap, crop_size, spNorm, model, opt, mse, nonorm):
    # img_pic_whole = cv2.imread(img_pic_whole_path)
    img_pic_align_crop_list, mat_list = simswap.get(img_pic_whole,crop_size)
    swap_result_list = []
    self_id_compare_values = [] 
    b_align_crop_tenor_list = []
    for b_align_crop in img_pic_align_crop_list:
        b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop,cv2.COLOR_BGR2RGB))[None,...].to(device)
        b_align_crop_tenor_arcnorm = spNorm(b_align_crop_tenor)
        b_align_crop_tenor_arcnorm_downsample = F.interpolate(b_align_crop_tenor_arcnorm, size=(112,112))
        b_align_crop_id_nonorm = model.netArc(b_align_crop_tenor_arcnorm_downsample.to(device))

        self_id_compare_values.append(mse(b_align_crop_id_nonorm, nonorm).detach().cpu().numpy())
        b_align_crop_tenor_list.append(b_align_crop_tenor)
    self_id_compare_values_array = np.array(self_id_compare_values) # 비슷한지 확인해서???
    self_min_index = np.argmin(self_id_compare_values_array) # 제일 작은게 그 사람이다???
    self_min_value = self_id_compare_values_array[self_min_index]

    if self_min_value < opt.id_thres:
        return b_align_crop_tenor_list[self_min_index], mat_list, self_min_index
    else:
        return None

def swap_deepfake(iscf, target_id_list, target_index, img_pic_whole, image_name, target_emb_name,
                  specific_person_align_crop, face_emb, target_hu_align_crop_tensor,mat_list, self_min_index,
                  model, opt, crop_size, spNorm, logoclass, nonorm, net):
    # if iscf == True: # closest인 경우
    #     n = 'closest'
    # else: n = 'furthest'

    swap_result_list = []
    b_align_crop_tenor_list = []
    for b_align_crop in specific_person_align_crop:
        swap_result, b_align_crop_tenor = process_crop(spNorm, model, mse,b_align_crop, nonorm)
        swap_result_list.append(swap_result)
        b_align_crop_tenor_list.append(b_align_crop_tenor)

    target_img_id = face_emb[target_id_list[target_index]].to(device)
    target_img_id_downsample = F.interpolate(target_img_id, size=(112,112))
    latend_id = model.netArc(target_img_id_downsample)
    latend_id = F.normalize(latend_id, p=2, dim=1)

    swap_result = model(None, target_hu_align_crop_tensor, latend_id, None, True)[0]
    # img_pic_whole = cv2.imread(path)
    return reverse2wholeimage(target_hu_align_crop_tensor, [swap_result], [mat_list[self_min_index]], crop_size, img_pic_whole, logoclass, \
        # os.path.join(opt.output_path, 'result_whole_swapspecific.jpg'), opt.no_simswaplogo,pasring_model =net,use_mask=opt.use_mask, norm = spNorm)
        './output.png', opt.no_simswaplogo, pasring_model = net,use_mask=opt.use_mask, norm = spNorm)
            
def get_emb(face_emb_dir, target='onlyGen'):
    face_emb = {}
    emb_ = {}
    if target == 'onlyGen':
        for emb_path in tqdm(glob.glob(face_emb_dir[0] + '*')):
            try:
                emb_ = np.load(emb_path, allow_pickle=True).item()
            except AttributeError:
                with open(file=emb_path, mode='rb') as f:
                    emb_ = pickle.load(f)
            if len(face_emb.keys()) == 0: face_emb = emb_
            else: face_emb.update(emb_)
    elif target == 'aihub':
        for emb_path in tqdm(glob.glob(face_emb_dir[1] + '/*')):
            try:
                emb_ = np.load(emb_path, allow_pickle=True).item()
            except AttributeError:
                with open(file=emb_path, mode='rb') as f:
                    emb_ = pickle.load(f)
            if len(face_emb.keys()) == 0: face_emb = emb_
            else: face_emb.update(emb_)
    elif target == 'cel':
        for emb_path in tqdm(glob.glob(face_emb_dir[2] + '/*')):
            try:
                emb_ = np.load(emb_path, allow_pickle=True).item()
            except AttributeError:
                with open(file=emb_path, mode='rb') as f:
                    emb_ = pickle.load(f)
            if len(face_emb.keys()) == 0: face_emb = emb_
            else: face_emb.update(emb_)
    elif target == 'oldaihub':
        for emb_path in tqdm(glob.glob(face_emb_dir[3] + '/*')):
            try:
                emb_ = np.load(emb_path, allow_pickle=True).item()
            except AttributeError:
                with open(file=emb_path, mode='rb') as f:
                    emb_ = pickle.load(f)
            if len(face_emb.keys()) == 0: face_emb = emb_
            else: face_emb.update(emb_)
    return face_emb


# CONFIGURE LOGGER
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('\x1b[33;20m[%(asctime)s - %(name)s - %(levelname)s] %(message)s\x1b[0m')
handler.setFormatter(formatter)
logger.addHandler(handler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CONFIGURE FLASK
app = Flask(__name__, static_folder = './../React_instagram_clone/build')
logger.info('Flask Server Loaded')

# CONFIGURE SIMSWAP
simswap, opt, crop_size, model, mse, spNorm, logoclass = config()
male_emb_dir = ['./data/M/', './data/test_M', './data/celebrity/Male', './data/zzold_aihub']
female_emb_dir = ['./data/W/', './data/test_W', './data/celebrity/Female', './data/zold_aihub']
pic_dir = './data/sample/'
specific_gender = 'M'
# target_emb_list = ['onlyGen','aihub','cel']#,'oldaihub']
target_emb = opt.target
num = 2
model = model.to(device)

male_face_emb = get_emb(male_emb_dir, target_emb)
male_face_emb = {k: v for k, v in male_face_emb.items() if int(k) in range(200)}
female_face_emb = get_emb(female_emb_dir, target_emb)
female_face_emb = {k: v for k, v in female_face_emb.items() if int(k) in range(200)}
male_id_list = list(male_face_emb.keys())
female_id_list = list(female_face_emb.keys())

print("Number of Target Embedding Image: ", len(male_id_list), len(female_id_list))

# ROUTING
@app.route('/', defaults = {'path': ''})
@app.route('/<path:path>')
def react_route(path):
    if path != '' and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/generate', methods = ['POST'])
def generate():
    req = request.json
    image = req['image']
    gender = req['gender']
    method = req['method']
    image = image.split(';')[1].split(',')[1]
    image = base64.b64decode(image)
    image = Image.open(BytesIO(image))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(f'Hello, Generate: option = {method}')

    if method == 'blur':
        bboxes, kpss = simswap.det_model.detect(image, threshold=simswap.det_thresh, max_num=0, metric='default')
        print(bboxes)

        blur_image = image.copy()

        # for index, (x1, y1, x2, y2, p) in enumerate(bboxes):
        x1, y1, x2, y2, p = list(bboxes)[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        roi = blur_image[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)

        blur_image[y1:y2, x1:x2] = blurred_roi

        blur_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)

        cv2.imwrite(f'blur_face.png', blur_image)
        blur_image = Image.fromarray(blur_image)

        buffer = BytesIO()
        blur_image.save(buffer, format = 'PNG')
        img_bytes = buffer.getvalue()
        encoded_str = base64.b64encode(img_bytes).decode('utf-8')
        
        return jsonify([encoded_str]), 200

    elif method == 'emoji':
        bboxes, kpss = simswap.det_model.detect(image, threshold=simswap.det_thresh, max_num=0, metric='default')
        emoji_images = []

        for index, emoji_filename in enumerate(os.listdir('./emojis')):
            emoji_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2RGBA)
            overlay = cv2.cvtColor(np.array(Image.open(f'./emojis/{emoji_filename}')), cv2.COLOR_RGBA2BGRA)

            for index2, (x1, y1, x2, y2, p) in enumerate(bboxes):
                lt_x = int((x1 + x2) / 2 - max(y2 - y1, x2 - x1) / 2)
                lt_y = int((y1 + y2) / 2 - max(y2 - y1, x2 - x1) / 2)
                rb_x = int((x1 + x2) / 2 + max(y2 - y1, x2 - x1) / 2)
                rb_y = int((y1 + y2) / 2 + max(y2 - y1, x2 - x1) / 2)
                
                overlay = cv2.resize(overlay, (rb_x - lt_x, rb_y - lt_y))
                alpha = overlay[:, :, 3]

                for j, y in enumerate(range(lt_y, rb_y)):
                    for i, x in enumerate(range(lt_x, rb_x)):
                        if alpha[j][i] != 0.0:
                            emoji_image[y][x] = overlay[j][i]

            cv2.imwrite(f'emoji_face_{index}.png', emoji_image)
            emoji_image = cv2.cvtColor(emoji_image, cv2.COLOR_BGR2RGB)
            emoji_images.append(Image.fromarray(emoji_image))

        image = choice(emoji_images)
        buffer = BytesIO()
        image.save(buffer, format = 'PNG')
        img_bytes = buffer.getvalue()
        encoded_str = base64.b64encode(img_bytes).decode('utf-8')

        return jsonify([encoded_str]), 200

    else:   # DeepPrivacy
        if gender == 'M':
            target_id_list = male_id_list
            face_emb = male_face_emb
        else:
            target_id_list = female_id_list
            face_emb = female_face_emb

        cv2.imwrite('test.png', image)

        align_crop, id_nonorm = get_embedding_specific_person(image, simswap, crop_size, model)

        # FIND INDICES
        idx = []
        
        id_compare_values_list = []
        for i in trange(0, len(target_id_list)):
            id_compare_values_list.append(compute_embedding_distance(face_emb, model, mse, id_nonorm, i, target_id_list))
        id_compare_values_array = np.array(id_compare_values_list)
        print(id_compare_values_array[:3])
        
        if method == 'closest':
            value = np.sort(id_compare_values_array)[:1]
        elif method == 'furthest':
            value = np.sort(id_compare_values_array)[-1:]

        idx = [np.where(id_compare_values_array == value[ii])[0][0] for ii in range(len(value))]
        
        print(f"Person ID:", idx)

        # SWAP
        if opt.use_mask:
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.to(device)
            save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net = None
        
        results = []

        target_hu_align_crop_tensor, mat_list, self_min_index = target_hu_inwhole(image, simswap, crop_size, spNorm, model, opt, mse, id_nonorm)
        if target_hu_align_crop_tensor == None:
            print('The person you specified is not found on the picture: {}'.format(image))
            # break
        else:
            for target_index in idx:
                result = swap_deepfake(True, target_id_list, target_index, image, 'result', target_emb, align_crop, face_emb, target_hu_align_crop_tensor, mat_list, self_min_index, model, opt, crop_size, spNorm, logoclass, id_nonorm, net)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                results.append(Image.fromarray(result))

        print('RESULT', results)

        time.sleep(1)

        encoded = dict()
        encoded_strs = []

        for image in results:
            buffer = BytesIO()
            image.save(buffer, format = 'PNG')
            img_bytes = buffer.getvalue()
            encoded_str = base64.b64encode(img_bytes).decode('utf-8')
            encoded_strs.append(encoded_str)

        # print(encoded)

        return jsonify(encoded_strs), 200

@app.route('/filter', methods = ['POST'])
def filter():
    req = request.json
    image = req['image']
    gender = req['gender']
    image = image.split(';')[1].split(',')[1]

    image = base64.b64decode(image)
    image = Image.open(BytesIO(image))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print('Hello, Generate!')
    if gender == 'M':
        target_id_list = male_id_list
        face_emb = male_face_emb
    else:
        target_id_list = female_id_list
        face_emb = female_face_emb

    align_crop, id_nonorm = get_embedding_specific_person(image, simswap, crop_size, model)

    # FIND INDICES
    idx = []
    
    id_compare_values_list = []
    for i in trange(0, len(target_id_list)):
        id_compare_values_list.append(compute_embedding_distance(face_emb, model, mse, id_nonorm, i, target_id_list))
    id_compare_values_array = np.array(id_compare_values_list)
    print(id_compare_values_array[:3])
    
    closest_value = np.sort(id_compare_values_array)[:10]
    furthest_value = np.sort(id_compare_values_array)[-10:]

    closest_idx = [np.where(id_compare_values_array == closest_value[ii])[0][0] for ii in range(len(closest_value))]
    furthest_idx = [np.where(id_compare_values_array == furthest_value[ii])[0][0] for ii in range(len(furthest_value))]

    # SWAP
    if opt.use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.to(device)
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net = None
    
    results = {
        'closest': [],
        'furthest': [],
        'others': []
    }

    target_hu_align_crop_tensor, mat_list, self_min_index = target_hu_inwhole(image, simswap, crop_size, spNorm, model, opt, mse, id_nonorm)
    if target_hu_align_crop_tensor == None:
        print('The person you specified is not found on the picture: {}'.format(image))
        # break
    else:
        for target_index in closest_idx:
            try:
                print(len(results['closest']))
                result = swap_deepfake(True, target_id_list, target_index, image, 'result', target_emb, align_crop, face_emb, target_hu_align_crop_tensor, mat_list, self_min_index, model, opt, crop_size, spNorm, logoclass, id_nonorm, net)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                results['closest'].append(Image.fromarray(result))
                Image.fromarray(result).save(f'{target_index}.png', dpi=(600, 600))
            except:
                continue
            finally:
                if len(results['closest']) == 3:
                    break

        for target_index in furthest_idx:
            try:
                result = swap_deepfake(True, target_id_list, target_index, image, 'result', target_emb, align_crop, face_emb, target_hu_align_crop_tensor, mat_list, self_min_index, model, opt, crop_size, spNorm, logoclass, id_nonorm, net)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                results['furthest'].append(Image.fromarray(result))
                Image.fromarray(result).save(f'{target_index}.png', dpi=(600, 600))
            except:
                continue
            finally:
                if len(results['furthest']) == 3:
                    break

    bboxes, kpss = simswap.det_model.detect(image, threshold=simswap.det_thresh, max_num=0, metric='default')

    blur_image = image.copy()

    x1, y1, x2, y2, p = list(bboxes)[1]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    roi = blur_image[y1:y2, x1:x2]
    blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)

    blur_image[y1:y2, x1:x2] = blurred_roi

    blur_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)

    # cv2.imwrite(f'blur_face.png', blur_image)
    blur_image = Image.fromarray(blur_image)
    blur_image.save('blur.png', dpi=(600, 600))

    results['others'].append(blur_image)

    emoji_images = []
    for index, emoji_filename in enumerate(os.listdir('./emojis')):
        emoji_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2RGBA)
        overlay = cv2.cvtColor(np.array(Image.open(f'./emojis/{emoji_filename}')), cv2.COLOR_RGBA2BGRA)

        # for index2, (x1, y1, x2, y2, p) in enumerate(bboxes):
        x1, y1, x2, y2, p = list(bboxes)[1]
        lt_x = int((x1 + x2) / 2 - max(y2 - y1, x2 - x1) / 2)
        lt_y = int((y1 + y2) / 2 - max(y2 - y1, x2 - x1) / 2)
        rb_x = int((x1 + x2) / 2 + max(y2 - y1, x2 - x1) / 2)
        rb_y = int((y1 + y2) / 2 + max(y2 - y1, x2 - x1) / 2)
        
        overlay = cv2.resize(overlay, (rb_x - lt_x, rb_y - lt_y))
        alpha = overlay[:, :, 3]

        for j, y in enumerate(range(lt_y, rb_y)):
            for i, x in enumerate(range(lt_x, rb_x)):
                if alpha[j][i] != 0.0:
                    emoji_image[y][x] = overlay[j][i]

        # cv2.imwrite(f'emoji_face_{index}.png', emoji_image)
        emoji_image = cv2.cvtColor(emoji_image, cv2.COLOR_BGR2RGB)
        results['others'].append(Image.fromarray(emoji_image))
        Image.fromarray(emoji_image).save(f'./emoji_{index}.png', dpi=(600, 600))

    time.sleep(1)

    encoded = dict()
    encoded_strs = []

    for category in results.keys():
        for image in results[category]:
            buffer = BytesIO()
            image.save(buffer, format = 'PNG')
            img_bytes = buffer.getvalue()
            encoded_str = base64.b64encode(img_bytes).decode('utf-8')
            encoded_strs.append(encoded_str)
        encoded[category] = encoded_strs

    return jsonify(encoded), 200

if __name__ == '__main__':
    warnings.simplefilter('ignore')
    app.run(host='0.0.0.0', port=8888, debug=True, use_reloader=False)