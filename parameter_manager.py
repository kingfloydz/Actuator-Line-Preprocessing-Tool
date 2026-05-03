import os
import sys
import shutil
import numpy as np
import stl


class ParameterManager:
    """参数管理类，通过BladeAnalyzer获取叶片参数，负责参数解析与模型数据加载"""
    
    def __init__(self, blade_analysis=None, processed_model_path=None, initial_setup=False):
        # 程序基础配置
        self.program_dir = self._get_program_directory()
        self.xfoil_workdir = os.path.join(self.program_dir, "Temp")
        self._create_work_directory()
        self.is_symmetric = True
        self.xfoil_core_lib_path = os.path.join(
            self.program_dir, "xfoil_core", "lib", "xfoil_core.dll"
        )
        
        # 尝试查找 xfoil.exe
        self.xfoil_exe_path = os.path.join(self.program_dir, "xfoil.exe")
        if not os.path.exists(self.xfoil_exe_path):
            # 尝试在兄弟目录查找
            sibling_path = os.path.join(os.path.dirname(self.program_dir), "使用xoil.exe", "xfoil.exe")
            if os.path.exists(sibling_path):
                self.xfoil_exe_path = sibling_path
            else:
                # 如果找不到，尝试在 Temp 目录查找（如果之前被复制过）
                temp_exe = os.path.join(self.xfoil_workdir, "xfoil.exe")
                if os.path.exists(temp_exe):
                    self.xfoil_exe_path = temp_exe
        
        # 关键修复：在所有情况下都定义参数文件路径
        self._param_file_path = os.path.join(self.program_dir, "parameter.dat")
        
        # 参数存储
        self.params = {}
        self.model_files = []  # 模型文件路径列表（仅单个桨叶模型）
        
        # 模型数据（按类型区分）
        self.mesh = []  # STL网格数据
        self.stl_vertices = []  # STL顶点 (n,3)
        self.stl_faces = []  # STL面 (n,3,3)
        self.obj_vertices = []  # OBJ顶点 (n,3)
        self.obj_faces = []  # OBJ面索引 (n,k)
        
        self.axial_direction = None
        
        self.rotor_speed_is_negative = False
        
        # 计算设置默认值
        self.default_settings = {
            '计算设置': {
                'num_sections': 50,
                'section_start_ratio': 0.15,
                'section_end_ratio': 0.9,
                'max_processes': 8,
                'num_foil_points': 250,
                'epsilon': None
            }
        }
        
        # 初始设置模式 - 仅获取原始模型文件
        if initial_setup:
            self._read_parameter_file()
            self.original_model_file = self._get_original_model_file()
            self.processed_model_prefix = os.path.join(self.program_dir, "processed_blade")
            
            self._parse_axial_direction()
            return
            
        # 从BladeAnalyzer获取的分析结果
        self.blade_analysis = blade_analysis
        
        # 读取并解析参数文件（包含轴向方向向量）
        self._read_parameter_file()
        self._parse_and_validate_parameters()
        
        # 加载处理后的单个桨叶模型
        self.model_files = [processed_model_path]
        self._load_model_files()
    
    def _get_program_directory(self):
        """获取程序所在目录"""
        try:
            if getattr(sys, 'frozen', False):
                return os.path.dirname(sys.executable)
            else:
                return os.path.dirname(os.path.abspath(__file__))
        except Exception as e:
            raise RuntimeError(f"获取程序目录失败: {str(e)}")
    
    def _create_work_directory(self):
        """创建工作目录"""
        try:
            if not os.path.exists(self.xfoil_workdir):
                os.makedirs(self.xfoil_workdir)
        except Exception as e:
            raise RuntimeError(f"创建工作目录失败: {str(e)}")
    
    def clear_work_directory(self):
        """清空工作目录"""
        try:
            if os.path.exists(self.xfoil_workdir):
                for item in os.listdir(self.xfoil_workdir):
                    item_path = os.path.join(self.xfoil_workdir, item)
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
        except Exception as e:
            raise RuntimeError(f"清空工作目录失败: {str(e)}")
    
    def _get_original_model_file(self):
        """获取原始模型文件路径"""
        if '模型文件' not in self.params:
            raise ValueError("参数文件缺少[模型文件]部分")
        model_sec = self.params['模型文件']
        if '文件1' not in model_sec:
            raise ValueError("模型文件部分缺少文件1（原始模型文件路径）")
        
        path = model_sec['文件1']
        if not os.path.isabs(path):
            path = os.path.join(self.program_dir, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"原始模型文件不存在: {path}")
        return path
    
    def _read_parameter_file(self):
        """读取parameter.dat文件（包含轴向方向向量）"""
        if not os.path.exists(self._param_file_path):
            raise FileNotFoundError(f"参数文件不存在: {self._param_file_path}")
        
        current_section = None
        with open(self._param_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('[') and line.endswith(']'):
                    current_section = line[1:-1].strip()
                    self.params[current_section] = {}
                elif '=' in line and current_section:
                    key, value = [part.strip() for part in line.split('=', 1)]
                    # 跳过将由BladeAnalyzer提供的其他几何参数，但保留轴向方向向量
                    if current_section == '几何参数' and key in [
                        '叶片数目', '质心坐标', 
                        '展向方向向量1', '展向方向向量2', '展向方向向量3'
                    ]:
                        continue
                    self.params[current_section][key] = value
                else:
                    raise ValueError(f"参数文件格式错误（第{line_num}行）: {line}")
        
        # 合并默认设置
        for section, defaults in self.default_settings.items():
            if section not in self.params:
                self.params[section] = defaults
            else:
                for key, default_val in defaults.items():
                    if key not in self.params[section]:
                        self.params[section][key] = default_val
    
    def _parse_axial_direction(self):
        """解析轴向方向向量（在初始设置模式和正常模式下均可使用）"""
        if '几何参数' not in self.params or '轴向方向向量' not in self.params['几何参数']:
            raise ValueError("参数文件[几何参数]部分缺少'轴向方向向量'")
        
        axial_str = self.params['几何参数']['轴向方向向量']
        try:
            self.axial_direction = np.array([float(x.strip()) for x in axial_str.split(',')], dtype=np.float64)
            if self.axial_direction.shape != (3,):
                raise ValueError("轴向方向向量必须包含3个分量")
            if np.linalg.norm(self.axial_direction) < 1e-10:
                raise ValueError("轴向方向向量不能为零向量")
            # 归一化处理
            self.axial_direction /= np.linalg.norm(self.axial_direction)
        except ValueError as e:
            raise ValueError(f"解析轴向方向向量失败: {str(e)}，格式应为'x,y,z'")
    
    def _parse_and_validate_parameters(self):
        """解析并验证所有参数（从dat文件读取轴向方向向量）"""
        # 从BladeAnalyzer获取其他几何参数
        if not self.blade_analysis:
            raise ValueError("未提供BladeAnalyzer的分析结果")
        
        # 1. 几何参数解析（部分来自BladeAnalyzer，轴向方向向量来自dat文件）
        self.num_blades = self.blade_analysis["num_blades"]
        self.centroid = self.blade_analysis["origin"]  # 质心坐标
        
        # 解析轴向方向向量（复用通用方法）
        self._parse_axial_direction()
        self.axis_width = self.blade_analysis["axis_width"]
        
        # 展向方向向量（来自BladeAnalyzer）
        self.span_directions = self.blade_analysis["span_directions"]
        self.is_symmetric = self.params['几何参数'].get('叶片是否对称', '否').lower() in ['是', 'true', '1']
        
        # 2. 空气属性解析
        air = self.params['空气属性']
        self.air_density = float(air['密度'])
        self.air_viscosity = float(air['动力粘度'])
        self.air_temperature = float(air['温度'])
        if self.air_density <= 0 or self.air_viscosity <= 0 or self.air_temperature <= 0:
            raise ValueError("空气属性必须为正数")
        
        # 3. 运行参数解析
        run = self.params['运行参数']
        # 解析转子转速：记录是否为负，并存储绝对值
        original_rotor_speed = float(run['转子转速'])
        self.rotor_speed_is_negative = original_rotor_speed < 0  # 记录转速是否为负
        self.rotor_speed = abs(original_rotor_speed)  # 仅存储绝对值
        
        self.inflow_velocity = np.array([float(x.strip()) for x in run['来流速度'].split(',')])
        if len(self.inflow_velocity) != 3:
            raise ValueError("来流速度格式错误，应为x,y,z")
        inflow_norm = np.linalg.norm(self.inflow_velocity)
        self.flow_direction_matches_axis = False
        if inflow_norm > 1e-10:
            inflow_dir = self.inflow_velocity / inflow_norm
            self.flow_direction_matches_axis = np.allclose(inflow_dir, self.axial_direction, atol=1e-6)
        
        # 4. 计算设置解析（已移除tail_threshold、leading_edge_weight、curvature_factor）
        calc = self.params['计算设置']
        self.num_sections = int(calc['num_sections'])
        self.section_start_ratio = float(calc['section_start_ratio'])
        self.section_end_ratio = float(calc['section_end_ratio'])
        self.max_processes = int(calc['max_processes'])
        self.num_foil_points = int(calc['num_foil_points'])
        
        # 验证计算设置（移除了相关参数的验证）
        if self.num_sections <= 0:
            raise ValueError("num_sections必须为正整数")
        if not (0 < self.section_start_ratio < self.section_end_ratio < 1):
            raise ValueError("section_start_ratio和section_end_ratio必须满足0 < start < end < 1")
        if self.max_processes < 1:
            raise ValueError("max_processes必须为正整数")
        if self.num_foil_points < 10:
            raise ValueError("num_foil_points必须至少为10")
    
    def _load_model_files(self):
        """加载处理后的单个桨叶模型文件"""
        for file in self.model_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"模型文件不存在: {file}")
            
            # 获取文件扩展名（小写）
            ext = os.path.splitext(file)[1].lower()
            
            if ext == '.stl':
                self._load_stl_file(file)
            elif ext == '.obj':
                self._load_obj_file(file)
            else:
                raise ValueError(f"不支持的文件类型: {ext}，仅支持.stl和.obj")
        
        # print(f"成功加载{len(self.model_files)}个模型文件（STL: {len(self.stl_vertices)}, OBJ: {len(self.obj_vertices)}）")
        self.vertices = self.stl_vertices + self.obj_vertices
        self.faces = self.stl_faces + self.obj_faces
    
    def _load_stl_file(self, file_path):
        """加载单个STL文件"""
        try:
            mesh = stl.mesh.Mesh.from_file(file_path)
            self.mesh.append(mesh)
            self.stl_vertices.append(mesh.vectors.reshape(-1, 3))
            self.stl_faces.append(mesh.vectors)
            # print(f"已加载STL文件: {os.path.basename(file_path)}")
        except Exception as e:
            raise RuntimeError(f"加载STL文件失败（{file_path}）: {str(e)}")

    def _load_obj_file(self, file_path):
        """加载单个OBJ文件"""
        try:
            vertices, faces = self._parse_obj_file(file_path)
            self.obj_vertices.append(vertices)
            self.obj_faces.append(faces)
            # print(f"已加载OBJ文件: {os.path.basename(file_path)}")
        except Exception as e:
            raise RuntimeError(f"加载OBJ文件失败（{file_path}）: {str(e)}")

    def _parse_obj_file(self, file_path):
        """解析OBJ文件，提取顶点和面对应关系"""
        vertices = []
        faces = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if not parts:
                    continue
                
                # 顶点数据 (v x y z)
                if parts[0] == 'v':
                    if len(parts) < 4:
                        continue  # 无效顶点行
                    try:
                        x, y, z = map(float, parts[1:4])
                        vertices.append([x, y, z])
                    except ValueError:
                        continue  # 忽略格式错误的顶点
                
                # 面数据 (f v1 v2 v3 ...)，只处理顶点索引
                elif parts[0] == 'f':
                    face_indices = []
                    for part in parts[1:]:
                        # OBJ索引格式可能为v/vt/vn，只取顶点索引部分
                        v_idx = part.split('/')[0]
                        try:
                            face_indices.append(int(v_idx))  # 保留1基索引
                        except ValueError:
                            break  # 忽略格式错误的面
                    if len(face_indices) >= 3:  # 只处理三角形及以上多边形
                        faces.append(face_indices)
        
        if not vertices:
            raise ValueError(f"OBJ文件{file_path}中未找到有效顶点数据")
        
        # 转换为numpy数组
        vertices_np = np.array(vertices, dtype=np.float64)
        faces_np = np.array(faces, dtype=object)  # 允许变长数组
        
        return vertices_np, faces_np
    