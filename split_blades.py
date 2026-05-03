import numpy as np
from stl import mesh
from collections import deque
import os

class BladeAnalyzer:
    def __init__(self, file_path, axial_direction=None):
        """初始化桨叶分析器，读取3D模型文件并准备基础数据"""
        self.file_path = file_path
        self.file_format = self._determine_file_format()
        
        # 核心数据存储
        self.vertices = None  # 顶点数据 (n, 3, 3)
        self.normals = None   # 法向量数据
        self.areas = None     # 每个面片的面积
        self.stl_mesh = None  # STL网格对象（仅STL文件有效）
        self.obj_vertices = None  # OBJ文件顶点列表
        
        # 分析结果数据（轴向从外部传入）
        self.principal_axis = axial_direction  # 从ParameterManager传入的轴向
        self.num_blades = None
        self.origin = None
        self.blade_regions = []  # 存储所有桨叶区域
        self.span_directions = []  # 存储所有桨叶展向（其他桨叶为[0,0,0]）
        
        # 初始化时读取3D模型文件
        self._read_3d_file()
    
    def _determine_file_format(self):
        """根据文件扩展名确定文件格式"""
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == '.stl':
            return 'stl'
        elif ext == '.obj':
            return 'obj'
        else:
            raise ValueError(f"不支持的文件格式: {ext}，仅支持STL和OBJ文件")
    
    def _read_3d_file(self):
        """根据文件格式读取相应的3D模型文件"""
        if self.file_format == 'stl':
            self._read_stl_file()
        elif self.file_format == 'obj':
            self._read_obj_file()
    
    def _compute_normals_and_areas(self, vertices):
        """计算法向量和面积（公共方法）"""
        v1 = vertices[:, 1] - vertices[:, 0]
        v2 = vertices[:, 2] - vertices[:, 0]
        normals = np.cross(v1, v2)
        areas = 0.5 * np.linalg.norm(normals, axis=1)
        return normals, areas
    
    def _read_stl_file(self):
        """读取STL文件，提取顶点坐标、法向量和每个面片的面积"""
        self.stl_mesh = mesh.Mesh.from_file(self.file_path)
        self.vertices = self.stl_mesh.vectors  # 形状: (n, 3, 3)
        self.normals, self.areas = self._compute_normals_and_areas(self.vertices)
    
    def _read_obj_file(self):
        """读取OBJ文件，提取顶点坐标并计算法向量和面积"""
        vertices = []  # 存储所有顶点
        faces = []     # 存储面的顶点索引
        
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                if parts[0] == 'v':
                    # 顶点坐标 (x, y, z)
                    x, y, z = map(float, parts[1:4])
                    vertices.append([x, y, z])
                elif parts[0] == 'f':
                    # 面数据，提取顶点索引
                    face_vertices = []
                    for p in parts[1:]:
                        # 转换为0-based索引
                        vertex_idx = int(p.split('/')[0]) - 1
                        face_vertices.append(vertex_idx)
                    
                    # 确保每个面都是三角形
                    if len(face_vertices) == 3:
                        faces.append(face_vertices)
                    elif len(face_vertices) > 3:
                        # 将多边形分解为三角形
                        for i in range(1, len(face_vertices)-1):
                            faces.append([face_vertices[0], face_vertices[i], face_vertices[i+1]])
        
        if not vertices or not faces:
            raise ValueError("OBJ文件中未找到有效的顶点或面数据")
        
        # 存储所有顶点并提取面片顶点
        self.obj_vertices = np.array(vertices, dtype=np.float64)
        self.vertices = np.array([[self.obj_vertices[i], self.obj_vertices[j], self.obj_vertices[k]] 
                                 for i, j, k in faces], dtype=np.float64)
        self.normals, self.areas = self._compute_normals_and_areas(self.vertices)
    
    def _get_plane_basis(self, axis):
        """获取垂直于给定轴的平面基向量（公共方法）"""
        if np.allclose(axis, [1, 0, 0]) or np.allclose(axis, [-1, 0, 0]):
            v1 = np.array([0, 1, 0])
        else:
            v1 = np.cross(axis, [1, 0, 0])
            v1 /= np.linalg.norm(v1) if np.linalg.norm(v1) > 1e-10 else 1
        
        v2 = np.cross(axis, v1)
        v2 /= np.linalg.norm(v2) if np.linalg.norm(v2) > 1e-10 else 1
        return v1, v2
    
    def project_to_plane(self, points, axis):
        """将三维点投影到垂直于给定轴的二维平面"""
        v1, v2 = self._get_plane_basis(axis)
        proj_matrix = np.array([v1, v2]).T
        return np.dot(points, proj_matrix)
    
    def _compute_weighted_origin(self):
        """计算所有三角面片的加权中心"""
        face_centers = np.mean(self.vertices, axis=1)
        return np.average(face_centers, axis=0, weights=self.areas)
    
    def _segment_blades(self):
        """分割模型，识别桨叶区域（返回所有可能的区域）"""
        face_centers = np.mean(self.vertices, axis=1)
        face_centers_shifted = face_centers - self.origin
        axis_coords = np.dot(face_centers_shifted, self.principal_axis)
        
        min_coord, max_coord = np.min(axis_coords), np.max(axis_coords)
        coord_range = max_coord - min_coord
        planes = np.linspace(min_coord - 0.1, max_coord + 0.1, 5) if np.isclose(coord_range, 0) else np.linspace(min_coord, max_coord, 10)

        all_proj = self.project_to_plane(face_centers_shifted, self.principal_axis)
        all_distances = np.linalg.norm(all_proj, axis=1)
        
        # 计算平面附近最小距离
        threshold = coord_range / 20 if not np.isclose(coord_range, 0) else 0.2
        
        # 生成30个均匀分布的方向（0到2π）
        angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))  # 形状为(30, 2)
        
        # 计算每个投影点的角度
        proj_angles = np.arctan2(all_proj[:, 1], all_proj[:, 0])  # 形状为(m,)
        proj_angles = (proj_angles + 2 * np.pi) % (2 * np.pi)  # 确保角度在0到2π之间
        
        min_distances = []
        for pc in planes:
            # 找到平面附近的面片
            mask = np.abs(axis_coords - pc) < threshold
            if not np.any(mask):
                continue  # 该平面附近没有面片，跳过
            
            # 获取这些面片的投影和角度
            plane_proj = all_proj[mask]
            plane_proj_angles = proj_angles[mask]
            
            # 为每个方向计算最远距离
            max_distances = []
            angle_threshold = np.pi / 30  # 约6度
            
            for i, angle in enumerate(angles):
                # 计算角度差
                angle_diff = np.abs(plane_proj_angles - angle)
                angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
                
                # 筛选角度接近的点
                angle_mask = angle_diff < angle_threshold
                
                if np.any(angle_mask):
                    dot_products = np.dot(plane_proj[angle_mask], directions[i])
                    max_distances.append(np.max(dot_products))
                else:
                    max_distances.append(0)
            
            max_distances = np.array(max_distances)
            
            # 过滤面片较少的平面
            if np.sum(max_distances <= 0) > 5:
                continue
            
            # 记录最小距离
            min_distances.append(np.min(max_distances))

        # 确定轴宽度
        max_min_dist = np.mean(all_distances) * 0.1 if not min_distances else np.max(min_distances)
        axis_width = 1.1 * max_min_dist
        
        # 识别桨叶区域（圆外面片）
        blade_mask = all_distances > axis_width
        blade_faces = np.where(blade_mask)[0]
        
        # 构建邻接列表
        num_faces = len(self.vertices)
        adjacency = [[] for _ in range(num_faces)]
        vertex_to_faces = {}
        vertices_rounded = np.round(self.vertices, 6)
        
        for i in range(num_faces):
            for v in vertices_rounded[i]:
                v_tuple = tuple(v)
                vertex_to_faces.setdefault(v_tuple, []).append(i)
        
        for faces in vertex_to_faces.values():
            for i in range(len(faces)):
                for j in range(i+1, len(faces)):
                    f1, f2 = faces[i], faces[j]
                    adjacency[f1].append(f2)
                    adjacency[f2].append(f1)
        
        # 去重邻接列表
        for i in range(num_faces):
            adjacency[i] = np.unique(adjacency[i])
        
        # 连通区域识别
        visited = np.zeros(num_faces, dtype=bool)
        blade_regions = []
        blade_mask_array = np.zeros(num_faces, dtype=bool)
        blade_mask_array[blade_faces] = True
        
        for face_idx in blade_faces:
            if not visited[face_idx]:
                queue = deque([face_idx])
                visited[face_idx] = True
                region = [face_idx]
                
                while queue:
                    current = queue.popleft()
                    for neighbor in adjacency[current]:
                        if blade_mask_array[neighbor] and not visited[neighbor]:
                            visited[neighbor] = True
                            region.append(neighbor)
                            queue.append(neighbor)
                
                blade_regions.append(region)
        
        # 过滤近原点区域
        threshold_distance = 0.1 * axis_width
        valid_regions = [r for r in blade_regions if np.min(all_distances[r]) > threshold_distance]
        return valid_regions, axis_width
    
    def _find_span_direction(self, region):
        """确定单个桨叶的展向方向"""
        region_vertices = self.vertices[region]
        region_areas = self.areas[region]
        face_centers = np.mean(region_vertices, axis=1)
        centered_centers = face_centers - self.origin
        projected_centers = self.project_to_plane(centered_centers, self.principal_axis)
        
        # 计算权重
        weights = region_areas / np.sum(region_areas)
        
        # 加权均值中心化
        weighted_mean = np.average(projected_centers, axis=0, weights=weights)
        centered_data = projected_centers - weighted_mean
        
        # 计算加权协方差矩阵
        weighted_data = centered_data * np.sqrt(weights)[:, np.newaxis]
        weighted_cov = np.dot(weighted_data.T, weighted_data)
        
        # 最大方差方向
        eigenvalues, eigenvectors = np.linalg.eig(weighted_cov)
        max_eigen_idx = np.argmax(eigenvalues)
        optimal_dir_2d = eigenvectors[:, max_eigen_idx].real  # 取实部
        
        # 调整方向
        mean_proj = np.average(projected_centers, axis=0, weights=region_areas)
        if np.dot(mean_proj, optimal_dir_2d) < 0:
            optimal_dir_2d = -optimal_dir_2d
        
        # 转换为三维方向
        v1, v2 = self._get_plane_basis(self.principal_axis)
        span_dir_3d = optimal_dir_2d[0] * v1 + optimal_dir_2d[1] * v2
        span_dir_3d /= np.linalg.norm(span_dir_3d) if np.linalg.norm(span_dir_3d) > 1e-10 else 1
        return span_dir_3d
    
    
    def export_blade(self, output_prefix, region_index=0):
        """导出指定桨叶区域为STL文件"""
        if not self.blade_regions or region_index >= len(self.blade_regions):
            raise ValueError("没有可导出的桨叶区域或区域索引无效")
        
        blade_region = self.blade_regions[region_index]
        blade_mesh = mesh.Mesh(np.zeros(len(blade_region), dtype=mesh.Mesh.dtype))
        blade_mesh.vectors = self.vertices[blade_region]
        blade_mesh.normals = self.normals[blade_region]
        blade_mesh.save(f"{output_prefix}_blade_{0}.stl")
    
    def run_analysis(self, output_prefix):
        """执行桨叶分析流程（使用从ParameterManager获取的轴向）"""
        if self.principal_axis is None:
            raise ValueError("轴向方向未设置，请从ParameterManager传入轴向")
        
        print(f"开始桨叶分析 (处理{self.file_format.upper()}文件)...")
        
        self.origin = self._compute_weighted_origin()

        
        # 获取所有可能的桨叶区域
        all_blade_regions, axis_width = self._segment_blades()
        self.num_blades = len(all_blade_regions)
        self.blade_regions = all_blade_regions
        
        if not all_blade_regions:
            print("未检测到桨叶，分析终止")
            return
        
        # 选择面积最大的桨叶区域
        region_areas = [np.sum(self.areas[region]) for region in all_blade_regions]
        max_area_idx = np.argmax(region_areas)
        
        # 计算所有桨叶的展向（仅最大面积桨叶计算实际展向，其他为[0,0,0]）
        self.span_directions = []
        # 先添加最大面积桨叶的展向（放到第一个位置）
        max_region = all_blade_regions[max_area_idx]
        max_span_dir = self._find_span_direction(max_region)
        self.span_directions.append(max_span_dir)


        # 再添加其他桨叶的展向（除了最大面积桨叶外，其余都设为[0,0,0]）
        for i, region in enumerate(all_blade_regions):
            if i != max_area_idx:
                self.span_directions.append(np.array([0, 0, 0]))


        # 导出面积最大的桨叶
        self.export_blade(output_prefix, max_area_idx)
        print(f"共检测到{self.num_blades}个桨叶")
        return {
            'num_blades': self.num_blades,
            "origin": self.origin, 
            "span_directions": self.span_directions, 
            "principal_axis": self.principal_axis,
            "axis_width": axis_width
        }
    