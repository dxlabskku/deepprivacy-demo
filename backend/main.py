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

# CONFIGURE LOGGER
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('\x1b[33;20m[%(asctime)s - %(name)s - %(levelname)s] %(message)s\x1b[0m')
handler.setFormatter(formatter)
logger.addHandler(handler)

device = torch.device('mps')

# CONFIGURE FLASK
app = Flask(__name__, static_folder = './../frontend/build')
logger.info('Flask Server Loaded')

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
    app.prepare(ctx_id=0, det_thresh=0.3, det_size=(640, 640), mode=mode)
    return app, opt, crop_size, model, mse, spNorm, logoclass

def compute_embedding_distance(asain_face_emb, model, mse, specific_person_id_nonorm, index,target_id_list):
    align_crop_tensor_arcnorm = asain_face_emb[target_id_list[index]].to(device)
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
                  specific_person_align_crop, asain_face_emb, target_hu_align_crop_tensor,mat_list, self_min_index,
                  model, opt, crop_size, spNorm, logoclass, nonorm, net):
    if iscf == True: # closest인 경우
        n = 'closest'
    else: n = 'furthest'

    output_file_name = f"deepfake_result/{image_name}_{target_emb_name}_{n}_{target_index}.jpg"
    print("Output File Name:",output_file_name)

    swap_result_list = []
    b_align_crop_tenor_list = []
    for b_align_crop in specific_person_align_crop:
        swap_result, b_align_crop_tenor = process_crop(spNorm, model, mse,b_align_crop, nonorm)
        swap_result_list.append(swap_result)
        b_align_crop_tenor_list.append(b_align_crop_tenor)

    target_img_id = asain_face_emb[target_id_list[target_index]].to(device)
    target_img_id_downsample = F.interpolate(target_img_id, size=(112,112))
    latend_id = model.netArc(target_img_id_downsample)
    latend_id = F.normalize(latend_id, p=2, dim=1)

    swap_result = model(None, target_hu_align_crop_tensor, latend_id, None, True)[0]
    # img_pic_whole = cv2.imread(path)
    return reverse2wholeimage(target_hu_align_crop_tensor, [swap_result], [mat_list[self_min_index]], crop_size, img_pic_whole, logoclass, \
        # os.path.join(opt.output_path, 'result_whole_swapspecific.jpg'), opt.no_simswaplogo,pasring_model =net,use_mask=opt.use_mask, norm = spNorm)
        os.path.join(opt.output_path, output_file_name), opt.no_simswaplogo,pasring_model = net,use_mask=opt.use_mask, norm = spNorm)
            
def get_emb(asain_face_emb_dir, target='onlyGen'):
    asain_face_emb = {}
    emb_ = {}
    if target == 'onlyGen':
        for emb_path in tqdm(glob.glob(asain_face_emb_dir[0] + '*')):
            try:
                emb_ = np.load(emb_path, allow_pickle=True).item()
            except AttributeError:
                with open(file=emb_path, mode='rb') as f:
                    emb_ = pickle.load(f)
            if len(asain_face_emb.keys()) == 0: asain_face_emb = emb_
            else: asain_face_emb.update(emb_)
    elif target == 'aihub':
        for emb_path in tqdm(glob.glob(asain_face_emb_dir[1] + '/*')):
            try:
                emb_ = np.load(emb_path, allow_pickle=True).item()
            except AttributeError:
                with open(file=emb_path, mode='rb') as f:
                    emb_ = pickle.load(f)
            if len(asain_face_emb.keys()) == 0: asain_face_emb = emb_
            else: asain_face_emb.update(emb_)
    elif target == 'cel':
        for emb_path in tqdm(glob.glob(asain_face_emb_dir[2] + '/*')):
            try:
                emb_ = np.load(emb_path, allow_pickle=True).item()
            except AttributeError:
                with open(file=emb_path, mode='rb') as f:
                    emb_ = pickle.load(f)
            if len(asain_face_emb.keys()) == 0: asain_face_emb = emb_
            else: asain_face_emb.update(emb_)
    elif target == 'oldaihub':
        for emb_path in tqdm(glob.glob(asain_face_emb_dir[3] + '/*')):
            try:
                emb_ = np.load(emb_path, allow_pickle=True).item()
            except AttributeError:
                with open(file=emb_path, mode='rb') as f:
                    emb_ = pickle.load(f)
            if len(asain_face_emb.keys()) == 0: asain_face_emb = emb_
            else: asain_face_emb.update(emb_)
    return asain_face_emb

# CONFIGURE SIMSWAP
male_emb_dir = ['/Volumes/DongJae_Full/data/M/', '/Volumes/DongJae_Full/data/test_M', '/Volumes/DongJae_Full/data/celebrity/Male', '/Volumes/DongJae_Full/data/zzold_aihub']
female_emb_dir = ['/Volumes/DongJae_Full/data/W/', '/Volumes/DongJae_Full/data/test_W', '/Volumes/DongJae_Full/data/celebrity/Female', '/Volumes/DongJae_Full/data/zold_aihub']
pic_dir = './data/sample/'
specific_gender = 'M'
target_emb_list = ['cel']#['onlyGen','aihub','cel']#,'oldaihub']
num = 2
simswap, opt, crop_size, model, mse, spNorm, logoclass = config()
model = model.to(device)

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
    image = request.json
    image = image.split(';')[1].split(',')[1]
    image = base64.b64decode(image)
    image = Image.open(BytesIO(image))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite('test.png', image)

    results = dict()

    align_crop, id_nonorm = get_embedding_specific_person(image, simswap, crop_size, model)

    for target_emb in target_emb_list:
        results[target_emb + '_closest'] = []
        results[target_emb + '_furthest'] = []

        # FIND INDICES
        closest_idx, furthest_idx = [], []

        print("Target Embbeding:", target_emb)
        if specific_gender == 'W':
            asain_face_emb = get_emb(female_emb_dir, target_emb)
        elif specific_gender == 'M':
            asain_face_emb = get_emb(male_emb_dir, target_emb)
        
        target_id_list = list(asain_face_emb.keys())
        print("Number of Target Embedding Image: ", len(target_id_list))

        id_compare_values_list = []
        for i in trange(0, len(target_id_list)):
            id_compare_values_list.append(compute_embedding_distance(asain_face_emb, model, mse, id_nonorm, i, target_id_list))
        id_compare_values_array = np.array(id_compare_values_list)
        print(id_compare_values_array[:3])
        
        closest_value = np.sort(id_compare_values_array)[:5]
        furthest_value = np.sort(id_compare_values_array)[-5:]

        closest_idx = [np.where(id_compare_values_array == closest_value[ii])[0][0] for ii in range(len(closest_value))]
        furthest_idx = [np.where(id_compare_values_array == furthest_value[ii])[0][0] for ii in range(len(furthest_value))]
        
        print(f"Closet person ID:", closest_idx)
        print(f"Furthest person ID:", furthest_idx)

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
        
        target_hu_align_crop_tensor,mat_list,self_min_index = target_hu_inwhole(image, simswap, crop_size, spNorm, model, opt, mse, id_nonorm)
        if target_hu_align_crop_tensor == None:
            print('The person you specified is not found on the picture: {}'.format(image))
            # break
        else:
            for target_index in closest_idx:
                result = swap_deepfake(True, target_id_list, target_index, image, 'result', target_emb, align_crop, asain_face_emb, target_hu_align_crop_tensor, mat_list, self_min_index, model, opt, crop_size, spNorm, logoclass, id_nonorm, net)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                results[target_emb + '_closest'].append(Image.fromarray(result))
            for target_index in furthest_idx:
                result = swap_deepfake(False, target_id_list, target_index, image, 'result', target_emb, align_crop, asain_face_emb, target_hu_align_crop_tensor, mat_list, self_min_index, model, opt, crop_size, spNorm, logoclass, id_nonorm, net)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                results[target_emb + '_furthest'].append(Image.fromarray(result))

    # print('RESULT', results)

    time.sleep(5)

    encoded = dict()
    for image_key in results.keys():
        encoded_strs = []

        for image in results[image_key]:
            buffer = BytesIO()
            image.save(buffer, format = 'PNG')
            img_bytes = buffer.getvalue()
            encoded_str = base64.b64encode(img_bytes).decode('utf-8')
            encoded_strs.append(encoded_str)
        
        encoded[image_key] = encoded_strs

    # print(encoded)

    return jsonify(encoded), 200

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5001, ssl_context = 'adhoc', debug = True)