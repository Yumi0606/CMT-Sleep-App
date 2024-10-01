# encoding: utf-8
import uuid
import time
import requests
from ai.auth_util import gen_sign_headers

import os
import json

# 请替换APP_ID、APP_KEY
APP_ID = '3035728524'
APP_KEY = 'OpHuKcIbKUpechKY'
URI = '/vivogpt/completions'
DOMAIN = 'api-ai.vivo.com.cn'
METHOD = 'POST'


def sync_vivogpt():
    params = {
        'requestId': str(uuid.uuid4())
    }
    print('requestId:', params['requestId'])

    # 获取 sync.py 文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取 sync.py 文件所在的目录路径
    current_dir_path = os.path.dirname(current_file_path)
    # 获取根目录路径（sync.py 文件位于根目录下的 ai 文件夹中，所以我们需要向上移动两级目录）
    root_path = os.path.join(current_dir_path, '..')
    # 构建到 sleep_data.json 文件的完整路径
    file_path = os.path.join(root_path, 'sleep_proportion', 'sleep_data.json')
    # 读取文件内容为字符串
    with open(file_path, 'r', encoding='utf-8') as file:
        str0 = file.read()
    # 将 JSON 字符串解析为 Python 对象
    data = json.loads(str0)
    # 打印解析后的数据
    sleep_data_str = ''
    for item in data:
        sleep_data_str += f"{item['name']}: {item['value']}min\n"
    # # 现在你可以使用 sleep_data_str 字符串
    # print(sleep_data_str)  # 打印字符串内容作为示例

    prompt0 = """
根据如下标准，诊断病人可能患有的睡眠障碍相关症状，并给出治疗建议.
睡眠分期时长的正常比例:
REM睡眠（快速眼动睡眠）：大约占整个睡眠周期的25%。
NREM睡眠（非快速眼动睡眠）：分为三个阶段，总共占睡眠周期的75%。其中,N1期：占5-10%。N2期：占45-55%。N3期：占15-25%。   
睡眠分期时长的比例异常可能的疾病:            
N3睡眠过少：可能导致身体恢复不足，影响记忆力和认知功能，可能与睡眠中断（如夜间醒来）或者发作性睡病有关。
REM睡眠过多：可能与睡眠呼吸暂停等睡眠障碍有关，这可能导致日间过度嗜睡和认知功能受损。

针对睡眠呼吸暂停综合症的治疗建议:
生活方式的改变：减重：对于超重或肥胖的患者，减重可以减少呼吸暂停的频率。改变睡姿：避免仰卧睡姿，可以使用侧睡辅助设备。避免酒精和某些药物：这些物质可能会放松喉咙肌肉，加重呼吸暂停。
持续正压呼吸机（CPAP）治疗：CPAP是治疗睡眠呼吸暂停的一线治疗方法，通过持续的正压保持呼吸道通畅。
口腔矫治器：对于轻中度睡眠呼吸暂停或CPAP不耐受的患者，口腔矫治器可以用来保持呼吸道开放。
手术治疗：对于某些患者，可能需要手术来移除或重塑阻塞气道的组织。

针对发作性睡病的治疗建议:
药物治疗：使用中枢神经系统兴奋剂（如哌甲酯）来提高日间觉醒水平。选择性5-羟色胺再摄取抑制剂（SSRIs）或其他药物来治疗猝倒症状。
生活方式的调整：规律的作息时间，包括规律的睡眠和短暂的日间小睡。避免可能导致猝倒的强烈情绪波动。
心理支持和教育：患者及其家人可能需要心理支持和教育，以更好地管理症状。
现有受试者睡眠分期时间比例如下,请给出可能患有的睡眠障碍相关症状和治疗建议。
        """
    prompt0 += sleep_data_str

    data = {
        'prompt': prompt0,
        'model': 'vivo-BlueLM-TB',
        'sessionId': str(uuid.uuid4()),
        'extra': {
            'temperature': 0.9
        }
    }
    headers = gen_sign_headers(APP_ID, APP_KEY, METHOD, URI, params)
    headers['Content-Type'] = 'application/json'

    start_time = time.time()
    url = 'https://{}{}'.format(DOMAIN, URI)
    response = requests.post(url, json=data, headers=headers, params=params)

    if response.status_code == 200:
        res_obj = response.json()
        print(f'response:{res_obj}')
        if res_obj['code'] == 0 and res_obj.get('data'):
            content = res_obj['data']['content']
            return content
    else:
        print(response.status_code, response.text)
    end_time = time.time()
    timecost = end_time - start_time
    print('请求耗时: %.2f秒' % timecost)



if __name__ == '__main__':
    sync_vivogpt()
