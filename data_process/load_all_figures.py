import json
from collections import defaultdict
import gzip
from tqdm import tqdm
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import amazon18_dataset2fullname
import requests

def download_image(url, save_path, timeout=15):
    """下载单张图片，失败时清理残留文件"""
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return True
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return False
    
def is_valid_jpg(jpg_file):

    with open(jpg_file, 'rb') as f:
        file_size = os.path.getsize(jpg_file)
        
        if file_size < 2:
            return False
        
        f.seek(file_size - 2)
        return f.read() == b'\xff\xd9'
        
        
def load_meta_items(file):
    items = {}
    with gzip.open(file, "r") as fp:
        for line in tqdm(fp, desc="Load metas"):
            data = json.loads(line)
            item = data["asin"]
            
            if 'imageURLHighRes' in data:
                imageURLHighRes = data['imageURLHighRes']
            else:
                imageURLHighRes = []
                
            # if len(imageURLHighRes) == 0:
            #     print(data)
            #     break

            items[item] = {'imageURLHighRes': imageURLHighRes}
            # break
            # print(items[item])
    return items


def load_meta_data(args):
    
    print('Process data: ')
    print(' Dataset: ', args.dataset)
    
    meta_data_path = args.meta_data_path

    dataset_full_name = amazon18_dataset2fullname[args.dataset]

    # load item IDs with meta data
    meta_file_path = os.path.join(meta_data_path, f'meta_{dataset_full_name}.json.gz')
    meta_items = load_meta_items(meta_file_path)
    
    return meta_items

def load_ratings_items(args, meta_items):
    # load ratings

    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    rating_file_path = os.path.join(args.rating_data_path, dataset_full_name + '.csv')

    filter_items = {}
    
    with open(rating_file_path, 'r') as fp:
        for line in tqdm(fp, desc='Load ratings'):
            try:
                item, user, rating, time = line.strip().split(',')
                
                if item in meta_items and item not in filter_items:
                    filter_items[item] = meta_items[item]
                    
            except ValueError:
                print(line)
                
    return filter_items

def load_5_core_review(args, meta_items):
    
    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    review_file_path = os.path.join(args.review_data_path, dataset_full_name + '_5.json.gz')

    filter_items = {}
    
    gin = gzip.open(review_file_path, 'rb')

    for line in tqdm(gin):

        line = json.loads(line)

        user_id = line['reviewerID']
        item_id = line['asin']
        time = line['unixReviewTime']

        if item_id in meta_items and item_id not in filter_items:
            filter_items[item_id] = meta_items[item_id]
                
    return filter_items

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def _download_one_item(asin, image_urls, save_path):
    """单个item的下载任务：按URL列表顺序尝试，成功一个即返回"""
    for image_url in image_urls:
        name = os.path.basename(image_url)
        save_file = os.path.join(save_path, name)

        if os.path.exists(save_file) and is_valid_jpg(save_file):
            return asin, [name]

        if download_image(image_url, save_file) and is_valid_jpg(save_file):
            return asin, [name]

    return asin, []


def main(args, meta_items):
    
    dataset = args.dataset
    save_path = f'{args.save_path}/{dataset}'

    item_images_file = f'{args.save_path}/{dataset}_images_info.json'
    os.makedirs(save_path, exist_ok=True)

    if os.path.exists(item_images_file):
        item_images = load_json(item_images_file)
    else:
        item_images = defaultdict(list)

    # 收集需要下载的任务（跳过已有结果的）
    tasks = {}
    skip_count = 0
    no_url_count = 0
    for asin, info in meta_items.items():
        if asin in item_images and len(item_images[asin]) != 0:
            skip_count += 1
            continue
        image_urls = info['imageURLHighRes']
        if len(image_urls) == 0:
            item_images[asin] = []
            no_url_count += 1
            continue
        tasks[asin] = image_urls

    print(f'已跳过(已下载): {skip_count}, 无图片URL: {no_url_count}, 待下载: {len(tasks)}')

    # 并行下载
    if tasks:
        ok_count = 0
        fail_count = 0
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(_download_one_item, asin, urls, save_path): asin
                for asin, urls in tasks.items()
            }
            with tqdm(total=len(futures), desc="并行下载图片") as pbar:
                for future in as_completed(futures):
                    asin, result = future.result()
                    item_images[asin] = result
                    if len(result) > 0:
                        ok_count += 1
                    else:
                        fail_count += 1
                    pbar.update(1)
        print(f'下载成功: {ok_count}, 下载失败: {fail_count}')

    miss_num = sum(1 for v in item_images.values() if len(v) == 0)
    print('miss num: ', miss_num)
    print("cover rate: ", (len(meta_items) - miss_num) / len(meta_items))
    with open(item_images_file, 'w', encoding='utf8') as f:
        json.dump(item_images, f, indent=4)


def parse_args():
    # 基于脚本位置计算项目默认路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_data_root = os.path.join(project_root, 'data', 'raw_amazon2018')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Arts', help='Instruments / Arts / Games')
    parser.add_argument('--meta_data_path', type=str, default=os.path.join(raw_data_root, 'Metadata'))
    parser.add_argument('--rating_data_path', type=str, default=os.path.join(raw_data_root, 'Ratings'))
    parser.add_argument('--review_data_path', type=str, default=os.path.join(raw_data_root, 'Review'))
    parser.add_argument('--save_path', type=str, default=os.path.join(raw_data_root, 'Images'))
    parser.add_argument('--max_workers', type=int, default=32, help='并行下载线程数')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    meta_items = load_meta_data(args)
    
    print('meta items: ', len(meta_items))
    
    # filter_items = load_ratings_items(args, meta_items)
    
    filter_items = load_5_core_review(args, meta_items)
    print('filter items: ', len(filter_items))

    for i in range(1):
        main(args, filter_items)