from scipy.spatial.transform import Rotation
import numpy  as np
import xml.etree.ElementTree as ET
import os
import json

def indent(elem, level=0):
    """用于递归地给 XML 添加缩进"""
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def get_camera_params(xml_input):
    """
    解析 MuJoCo XML。如果缺少必要的属性，程序将直接中止并报错。
    """
    # 1. 判断并加载 XML
    if os.path.isfile(xml_input):
        tree = ET.parse(xml_input)
        root = tree.getroot()
    else:
        root = ET.fromstring(xml_input)
        
    cameras = {}
    
    # 2. 查找所有相机
    for cam in root.iter('camera'):
        # 直接使用 cam.attrib['key']，如果 key 不存在会抛出 KeyError 导致程序中止
        name = cam.attrib['name']
        pos_str = cam.attrib['pos']
        xyaxes_str = cam.attrib['xyaxes']
        
        # 转换数据类型，如果 split 后长度不对或内容不是数字，会抛出 ValueError
        pos = [float(x) for x in pos_str.split()]
        xyaxes = [float(x) for x in xyaxes_str.split()]
        
        # 严格检查长度（MuJoCo 规范：pos 为 3，xyaxes 为 6）
        if len(pos) != 3:
            raise ValueError(f"相机 '{name}' 的 pos 属性必须包含 3 个数值，当前为: {len(pos)}")
        if len(xyaxes) != 6:
            raise ValueError(f"相机 '{name}' 的 xyaxes 属性必须包含 6 个数值，当前为: {len(xyaxes)}")
        
        cameras[name] = {
            'pos': pos,
            'xyaxes': xyaxes
        }
        
    return cameras

def save_camera_params_to_xml(input_xml_path, output_xml_path, cameras_dict):
    """
    读取现有的 XML 文件，根据 cameras_dict 更新相机参数，并保存到新文件中。
    
    参数:
    input_xml_path: 原始 XML 文件路径
    output_xml_path: 保存更新后的 XML 文件路径
    cameras_dict: 格式为 {'cam_name': {'pos': [...], 'xyaxes': [...]}} 的字典
    """
    # 1. 解析原始文件
    tree = ET.parse(input_xml_path)
    root = tree.getroot()
    
    # 2. 遍历字典中的每一个相机数据
    for cam_name, params in cameras_dict.items():
        # 严格检查字典内部数据长度
        if len(params['pos']) != 3:
            raise ValueError(f"相机 '{cam_name}' 的 pos 长度错误")
        if len(params['xyaxes']) != 6:
            raise ValueError(f"相机 '{cam_name}' 的 xyaxes 长度错误")
            
        # 3. 在 XML 中查找具有匹配 name 属性的 camera 节点
        # 使用 XPath 语法查找: .//camera[@name='xxx']
        cam_node = root.find(f".//camera[@name='{cam_name}']")
        
        if cam_node is None:
            raise KeyError(f"在 XML 文件中未找到名为 '{cam_name}' 的相机节点")
        
        # 4. 将列表转换回空格分隔的字符串
        pos_str = " ".join(map(str, params['pos']))
        xyaxes_str = " ".join(map(str, params['xyaxes']))
        
        # 5. 更新节点属性
        cam_node.set('pos', pos_str)
        cam_node.set('xyaxes', xyaxes_str)
    
    # 6. 将修改后的树写回文件
    # encoding="utf-8" 确保字符正确，xml_declaration=True 保留文件头
    indent(root) # 缩进美化
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    print(f"成功更新并保存至: {output_xml_path}")

def xyaxes_to_quat(xyaxes):
    """
    将 MuJoCo 的 xyaxes 转换为四元数 [w, x, y, z]
    xyaxes: 长度为 6 的列表或数组 [x1, x2, x3, y1, y2, y3]
    """
    x = np.array(xyaxes[:3])
    y = np.array(xyaxes[3:])
    
    # 1. 归一化 X 轴
    x /= np.linalg.norm(x)
    # 2. 计算 Z 轴 (Z = X cross Y) 并归一化
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    # 3. 重新计算 Y 轴以确保严格正交 (Y = Z cross X)
    y = np.cross(z, x)
    
    # 4. 构建旋转矩阵 (列向量分别为 x, y, z)
    mat = np.stack([x, y, z], axis=1)
    
    # 5. 转换为四元数 (scipy 默认是 [x, y, z, w])
    quat_xyzw = Rotation.from_matrix(mat).as_quat()
    
    # 6. 调整为 MuJoCo 的 [w, x, y, z] 格式
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

def quat_to_xyaxes(quat):
    """
    将四元数 [w, x, y, z] 转换为 MuJoCo 的 xyaxes
    """
    # 1. 将 MuJoCo 格式 [w, x, y, z] 转为 scipy 格式 [x, y, z, w]
    quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]
    
    # 2. 转换为旋转矩阵
    mat = Rotation.from_quat(quat_xyzw).as_matrix()
    
    # 3. 提取前两列 (X 轴 和 Y 轴)
    x_axis = mat[:, 0]
    y_axis = mat[:, 1]
    
    # 4. 拼接并展平
    return np.concatenate([x_axis, y_axis])

def scale_distance_from_pivot(original_quat=None, original_pos=None, scale_factor=1.3, pivot_point=None):
    """
    计算位置点到 (0, 0, 0.8) 的距离并按比例缩放，保持四元数不变
    
    参数:
        original_quat: 原始四元数 [w, x, y, z] (可选)
        original_pos: 原始位置 [x, y, z] (可选)
        scale_factor: 距离缩放因子 (默认1.5)
        pivot_point: 球面旋转中心[x,y,z]
        
    返回:
        字典包含:
        - 'new_quat': 原始四元数 [w, x, y, z] (如果输入了original_quat)
        - 'new_pos': 缩放后的位置 [x, y, z] (如果输入了original_pos)
    """
    result = {}
    
    # 定义轴点
    assert pivot_point is not None, "错误：pivot_point 尚未定义，请检查初始化逻辑。"
    pivot_point = np.array(pivot_point)
    
    # 处理四元数（保持不变）
    if original_quat is not None:
        result['new_quat'] = original_quat
    
    # 处理位置点缩放
    if original_pos is not None:
        original_pos = np.array(original_pos)
        # 计算从轴点到原始位置的向量
        vec_to_point = original_pos - pivot_point
        # 缩放这个向量
        scaled_vec = vec_to_point * scale_factor
        # 计算新位置
        new_pos = pivot_point + scaled_vec
        result['new_pos'] = new_pos.tolist() if isinstance(new_pos, np.ndarray) else new_pos
    
    return result

def rotate_around_y(original_quat=None, original_pos=None, degrees=0, pivot_point=None):
    """
    计算四元数和/或3D位置点绕自定义轴 (x=0, z=0.8) 的平行于Y轴的轴旋转指定角度后的新值
    
    参数:
        original_quat: 原始四元数 [w, x, y, z] (可选)
        original_pos: 原始位置 [x, y, z] (可选)
        degrees: 旋转角度（度数），正值为从X轴向Z轴旋转方向
        pivot_point: 球面旋转中心[x,y,z]
        
    返回:
        字典包含:
        - 'new_quat': 旋转后的四元数 [w, x, y, z] (如果输入了original_quat)
        - 'new_pos': 旋转后的位置 [x, y, z] (如果输入了original_pos)
    """
    result = {}
    
    # 定义旋转轴 (x=0, z=0.8) 的平行于Y轴的向量
    axis = np.array([0, 1, 0])  # 方向与Y轴相同
    assert pivot_point is not None, "错误：pivot_point 尚未定义，请检查初始化逻辑。"
    axis_point = np.array(pivot_point)  # 轴经过的点
    
    # 创建绕自定义轴的旋转
    custom_rotation = Rotation.from_rotvec(np.radians(-degrees) * axis)
    
    # 处理四元数旋转
    if original_quat is not None:
        original_rot = Rotation.from_quat([original_quat[1], original_quat[2], original_quat[3], original_quat[0]])
        combined_rot = custom_rotation * original_rot
        new_quat = combined_rot.as_quat()
        result['new_quat'] = [float(new_quat[3]), float(new_quat[0]), float(new_quat[1]), float(new_quat[2])]
    
    # 处理位置点旋转
    if original_pos is not None:
        # 对于点旋转，需要先平移到旋转轴，旋转后再平移回来
        translated_pos = np.array(original_pos) - axis_point
        rotated_pos = custom_rotation.apply(translated_pos)
        final_pos = rotated_pos + axis_point
        result['new_pos'] = final_pos.tolist() if isinstance(final_pos, np.ndarray) else final_pos
    
    return result

def rotate_around_z(original_quat=None, original_pos=None, degrees=0, pivot_point=None):
    """
    计算四元数和/或3D位置点绕Z轴旋转指定角度后的新值
    
    参数:
        original_quat: 原始四元数 [w, x, y, z] (可选)
        original_pos: 原始位置 [x, y, z] (可选)
        degrees: 旋转角度（度数），正值为逆时针方向
        pivot_point: 球面旋转中心[x,y,z]
    返回:
        字典包含:
        - 'new_quat': 旋转后的四元数 [w, x, y, z] (如果输入了original_quat)
        - 'new_pos': 旋转后的位置 [x, y, z] (如果输入了original_pos)
    """
    result = {}
    z_rotation = Rotation.from_euler('z', degrees, degrees=True)
    
    if original_quat is not None:
        original_rot = Rotation.from_quat([original_quat[1], original_quat[2], original_quat[3], original_quat[0]])
        combined_rot = z_rotation * original_rot
        new_quat = combined_rot.as_quat()
        result['new_quat'] = [float(new_quat[3]), float(new_quat[0]), float(new_quat[1]), float(new_quat[2])]
    
    if original_pos is not None:
        # 修正：相对于 pivot_point 进行旋转
        assert pivot_point is not None, "错误：pivot_point 尚未定义，请检查初始化逻辑。"
        pivot = np.array(pivot_point) if pivot_point is not None else np.array([0, 0, 0])
        translated_pos = np.array(original_pos) - pivot # 移动到以 pivot 为原点的空间
        rotated_pos = z_rotation.apply(translated_pos) # 执行旋转
        final_pos = rotated_pos + pivot                # 移回原位
        result['new_pos'] = final_pos.tolist()
    
    return result

def get_lookat_quat(cam_pos, pivot_point=None):
    """
    计算使相机位置 cam_pos 对准 pivot_point 的四元数 [w, x, y, z] (MuJoCo 格式)
    假设世界坐标系的 Up 轴为 [0, 0, 1]
    """
    cam_pos = np.array(cam_pos)
    assert pivot_point is not None, "错误：pivot_point 尚未定义，请检查初始化逻辑。"
    pivot_point = np.array(pivot_point)

    # 1. Z轴 (Forward): 从目标指向相机 (因为 MuJoCo 相机看向 -Z，所以 +Z 指向后方)
    z_axis = cam_pos - pivot_point
    norm_z = np.linalg.norm(z_axis)
    if norm_z < 1e-6:
        # 如果相机和目标重合，返回默认朝向 (避免除以0)
        return np.array([1.0, 0.0, 0.0, 0.0])
    z_axis /= norm_z

    # 2. X轴 (Right): World_Up cross Z_axis
    world_up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(world_up, z_axis)
    norm_x = np.linalg.norm(x_axis)
    
    # 处理 Gimbal Lock (当相机在目标正上方或正下方时)
    if norm_x < 1e-6:
        # 此时 Z 轴平行于 World Up，我们假设 X 轴指向世界 X
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis /= norm_x

    # 3. Y轴 (Up): Z_axis cross X_axis (确保正交)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # 4. 构建旋转矩阵 [x, y, z] (列向量)
    mat = np.stack([x_axis, y_axis, z_axis], axis=1)

    # 5. 转换为四元数
    # scipy 返回 [x, y, z, w]
    quat_xyzw = Rotation.from_matrix(mat).as_quat()
    
    # 6. 转为 MuJoCo [w, x, y, z]
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

def new_perspective(pivot_point, pos_av, quat_av,  horizon_view = 0, up_view = 0, scale_factor = 1.0, end_point_rot = 0, end_point_vertical = 0, look_at_center = False):
    """ 在mujoco世界坐标系下做相机的位姿变化, 相机以世界坐标系为中心做球面旋转, 由5个参数控制球面旋转, 现在horizon_view，end_point_rot是右手定则，其他是左手定则

    Args:
        pivot_point (list): 球面旋转中心，形式为[x,y,z]
        pos_av (list): 原始相机在世界坐标系下的位置, 形式为(x,y,z)
        quat_av (list): 原始相机在世界坐标系下的姿态，形式为(w, x, y, z), 这也是mujoco的四元素表示惯例, 实部在前
        horizon_view (float): 水平旋转角度, Defaults to 0.
        up_view (float): 垂直旋转角度,注意这里是绕着过旋转中心并且与世界y轴平行的轴进行垂直旋转. Defaults to 0.
        scale_factor (float): 相机沿着世界坐标系原点指向相机光心的射线移动的比例，控制相机远近， Defaults to 1.0.
        end_point_rot (float): 相机的水平偏航变化角度. Defaults to 0.
        end_point_vertical (float): 相机的垂直俯仰变化角度. Defaults to 0.
        look_at_center (bool): 是否强制让相机光轴对准旋转中心(0,0,0.8). Defaults to True.
    Return:
        pos_view, quat_view: 两个列表,[x,y,z],[w,x,y,z]
    """
    assert pivot_point is not None, "错误：pivot_point 尚未定义，请检查初始化逻辑。"
    

    # 1. 执行位置变换 (球面旋转)
    # ----------------------------------------------------------------
    # 这里我们主要关心 pos_view 的变化，quat_view 如果开启 look_at 将会被覆盖
    if int(up_view) != 0:
        result_up = rotate_around_y(original_quat=quat_av, original_pos=pos_av, degrees=int(up_view),pivot_point=pivot_point)
        pos_up = result_up['new_pos']
        # 级联旋转
        result_view = rotate_around_z(original_quat=result_up['new_quat'], original_pos=pos_up, degrees=int(horizon_view), pivot_point=pivot_point)
        pos_view = result_view['new_pos']
        quat_view = result_view['new_quat'] # 暂存
    else:
        result_view = rotate_around_z(original_quat=quat_av, original_pos=pos_av, degrees=int(horizon_view), pivot_point=pivot_point)
        pos_view = result_view['new_pos']
        quat_view = result_view['new_quat'] # 暂存

    # 2. 执行缩放 (Scale)
    # ----------------------------------------------------------------
    if float(scale_factor) != 1.0:
        result = scale_distance_from_pivot(original_quat=quat_view, original_pos=pos_view, scale_factor=float(scale_factor),pivot_point=pivot_point)
        pos_view = result['new_pos']
        # quat 保持不变

    # 3. 核心修改：应用 LookAt
    # ----------------------------------------------------------------
    if look_at_center:
        # 强制计算一个新的四元数，使相机看向 pivot_point
        # 注意：这里会覆盖掉之前通过 rotate_around 计算出的旋转累积（这是正确的，因为我们要对准中心）
        print("强制光轴对齐已开启")
        quat_view = get_lookat_quat(pos_view, pivot_point=pivot_point).tolist()

    # 4. 执行末端微调 (End Point Rotation)
    # ----------------------------------------------------------------
    # 这些旋转是在 LookAt 基础上的局部微调 (如由 LookAt 定好基准后，再稍微抬头或偏头)
    if int(end_point_rot) != 0:
        # 绕 Z 轴 (Roll/Yaw 取决于当前坐标系，通常用于画面倾斜或偏航)
        result_view = rotate_around_z(original_quat=quat_view, degrees=int(end_point_rot))
        quat_view = result_view['new_quat']
    
    if int(end_point_vertical) != 0:
        # 绕 Y 轴 (Pitch，抬头低头)
        result_view = rotate_around_y(original_quat=quat_view, degrees=int(end_point_vertical),pivot_point=pivot_point)
        quat_view = result_view['new_quat']

    # 格式化输出 (保留4位小数)
    pos_view = [round(x, 4) for x in pos_view]
    quat_view = [round(x, 4) for x in quat_view]

    print(f"pos=[{','.join(map(str, pos_view))}],")
    print(f"quat=[{','.join(map(str, quat_view))}]")
    
    return pos_view, quat_view

# SMALL只进行水平球面旋转，范围在[-5,5]度
SMALL = {
    "right0":{
        "horizon_view":5,
        "up_view": 0,
        "scale_factor": 1.0,
        "look_at_center": False,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "right1":{
        "horizon_view":-5,
        "up_view": 0,
        "scale_factor": 1.0,
        "look_at_center": False,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "left0":{
        "horizon_view":-5,
        "up_view": 0,
        "scale_factor": 1.0,
        "look_at_center": False,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "left1":{
        "horizon_view":5,
        "up_view": 0,
        "scale_factor": 1.0,
        "look_at_center": False,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "forward":{
        "horizon_view":5,
        "up_view": 0,
        "scale_factor": 1.0,
        "look_at_center": False,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    
}
# MEDIUM
MEDIUM = {
    "right0":{
        "horizon_view":-10,
        "up_view": -5,
        "scale_factor": 1.0,
        "look_at_center": False,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "right1":{
        "horizon_view":-10,
        "up_view": 5,
        "scale_factor": 0.9,
        "look_at_center": False,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "left0":{
        "horizon_view":10,
        "up_view": 5,
        "scale_factor": 1.0,
        "look_at_center": False,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "left1":{
        "horizon_view":10,
        "up_view": -5,
        "scale_factor": 0.9,
        "look_at_center": False,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "forward":{
        "horizon_view":-10,
        "up_view": -5,
        "scale_factor": 0.9,
        "look_at_center": False,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    
}
LARGE = {
    "right0":{
        "horizon_view":-15,
        "up_view": -10,
        "scale_factor": 1.2,
        "look_at_center": False,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "right1":{
        "horizon_view":-15,
        "up_view": 10,
        "scale_factor": 0.8,
        "look_at_center": False,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "left0":{
        "horizon_view":-15,
        "up_view": -10,
        "scale_factor": 1.2,
        "look_at_center": False,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "left1":{
        "horizon_view":-15,
        "up_view": 10,
        "scale_factor": 0.8,
        "look_at_center": False,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "forward":{
        "horizon_view":-15,
        "up_view": -10,
        "scale_factor": 0.8,
        "look_at_center": False,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    
}
# ALIGN
ALIGN = {
    "right0":{
        "horizon_view":0,
        "up_view": 0,
        "scale_factor": 1.0,
        "look_at_center": True,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "right1":{
        "horizon_view":-0,
        "up_view": -0,
        "scale_factor": 1.0,
        "look_at_center": True,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "left0":{
        "horizon_view":-0,
        "up_view": -0,
        "scale_factor": 1.0,
        "look_at_center": True,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "left1":{
        "horizon_view":0,
        "up_view": 0,
        "scale_factor": 1.0,
        "look_at_center": True,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    "forward":{
        "horizon_view":-0,
        "up_view": 0,
        "scale_factor": 1.0,
        "look_at_center": True,
        "end_point_rot":0,
        "end_point_vertical":0
    },
    
}
if __name__ == "__main__":
    
    camera_perturbation = "large"
    pivot_point = [0, -0.02, 1.196] #这是目前我们定义的场景中心的世界坐标
    
    imput_camera_xml_path = "VLABench/assets/base/5_cameras.xml"
    output_camera_xml_path = f"VLABench/assets/base/5_cameras_{camera_perturbation}_perturbation.xml"
    output_camera_control_perams_path = f"./work/camera_control_params/5_cameras_{camera_perturbation}_perturbation.json"
    # 读取camera xml文件
    cameras_params = get_camera_params(xml_input=imput_camera_xml_path)

    new_cameras_params = {}
    if camera_perturbation == "small": 
        control_params = SMALL
    elif camera_perturbation =="medium":
        control_params = MEDIUM
    elif camera_perturbation =="align":
        control_params = ALIGN
    else:
        control_params = LARGE
    
    for name, params in cameras_params.items():
        pos_av = params["pos"]
        quat_av = xyaxes_to_quat(params["xyaxes"]).tolist()
        pos_view, quat_view  = new_perspective(pivot_point,
                                               pos_av, quat_av, 
                                               control_params[name]["horizon_view"], 
                                               control_params[name]["up_view"], 
                                               control_params[name]["scale_factor"], 
                                               control_params[name]["end_point_rot"], 
                                               control_params[name]["end_point_vertical"],
                                               control_params[name]["look_at_center"]) #  <--- 是否开启光轴对齐功能
        new_cameras_params[name] = {
            'pos': pos_view,
            'xyaxes': quat_to_xyaxes(quat_view)
        }
    
    print("new camera params: ",new_cameras_params)

    save_camera_params_to_xml(imput_camera_xml_path, output_camera_xml_path, new_cameras_params)
    
    
    control_params["imput_camera_xml_path"] = imput_camera_xml_path
    control_params["pivot_point"] = pivot_point
    with open(output_camera_control_perams_path,"w") as f:
        json.dump(control_params, f, indent=4)
    print(f"control_params are save to {output_camera_control_perams_path}")

    
        
        
        

