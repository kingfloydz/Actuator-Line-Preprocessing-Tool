import numpy as np
import time
from collections import defaultdict
from parameter_manager import ParameterManager

class GeometryAnalyzer:
    """STL/OBJ模型几何处理模块，先筛选截面再计算特征"""
    def __init__(self, param_manager):
        self.param_manager = param_manager
        # 预计算常用常量
        self.epsilon = 1e-8
        self.epsilon_sq = self.epsilon **2

    def initialize_from_param_manager(self):
        """从ParameterManager初始化几何分析器"""
        # 基础参数初始化
        self.is_symmetric = self.param_manager.is_symmetric
        self.num_blades = self.param_manager.num_blades
        self.sym_num_blades = self.num_blades if self.is_symmetric else None
        if self.is_symmetric:
            self.num_blades = 1 
        self.centroid = self.param_manager.centroid
        self.axial_direction = self.param_manager.axial_direction
        # 预计算单位化的轴向向量
        self.axial_unit = self.axial_direction / np.linalg.norm(self.axial_direction)
        self.span_directions = self.param_manager.span_directions
        # 预计算单位化的展向向量
        self.span_units = [dir for dir in self.span_directions]
        self.rotor_speed_is_negative = self.param_manager.rotor_speed_is_negative
        self.model_files = self.param_manager.model_files        
        self.mesh = self.param_manager.mesh  # 统一的网格数据
        self.vertices = self.param_manager.vertices  # 统一的顶点数据
        self.faces = self.param_manager.faces        # 统一的面数据
        self.axis_width = self.param_manager.axis_width
        # 初始化叶片分析相关属性
        self.blade_indices = []
        self.blade_spans = []
        self.blade_roots = []
        self.all_sections = []  # 存储所有生成的截面
        self.blade_sections = []  # 存储筛选后的截面
        self.interval = None
        self.segments_per_blade = None  # 每片叶片的段数

    def analyze_blade_geometry(self):
        """分析叶片几何特征，计算展长与叶尖位置"""
        # 初始化叶片索引
        for i in range(self.num_blades if not self.is_symmetric else 1):
            self.blade_indices.append(np.arange(len(self.mesh[i].vectors)))

        self.blade_spans = []

        # 遍历叶片计算展长
        for i, indices in enumerate(self.blade_indices if not self.is_symmetric else [self.blade_indices[0]] * self.num_blades):
            blade_mesh = self.mesh[i % len(self.mesh)]
            blade_vertices = blade_mesh.vectors[indices].reshape(-1, 3)

            # 使用向量化计算投影
            centroid_diff = blade_vertices - self.centroid
            projections = np.dot(centroid_diff, self.span_units[i])

            max_idx = np.argmax(projections)
            max_projection = projections[max_idx]
            
            # 计算根部坐标
            blade_root = self.centroid + self.span_units[i] * (self.axis_width + 0.01 * max_projection)
            
            # 叶尖坐标
            blade_tip = blade_vertices[max_idx]
            # 展长为叶尖与根部在展向的投影差
            span_length = max_projection - np.dot(blade_root - self.centroid, self.span_units[i])
            self.blade_spans.append(span_length)
            self.blade_roots.append(blade_root)

            print(f"叶片 {i+1}:  根部坐标: {blade_root}  叶尖坐标: {blade_tip}  展长: {span_length:.4f} m")

    def generate_section_plans(self):
        """生成叶片展向分段方案"""
        if not self.blade_spans:
            raise ValueError("请先调用analyze_blade_geometry分析叶片几何特征")

        # 从ParameterManager获取参数
        start_ratio = self.param_manager.section_start_ratio
        end_ratio = self.param_manager.section_end_ratio
        self.segments_per_blade = self.param_manager.num_sections  # 段数
        num_per_segment = 5  # 每段的截面数

        self.all_sections = []  # 存储所有可能的截面用于筛选
        for i in range(self.num_blades):
            blade_root = self.blade_roots[i]
            span_unit = self.span_units[i]
            span_length = self.blade_spans[i]

            # 计算每段的比例长度
            total_ratio_range = end_ratio - start_ratio
            segment_ratio_length = total_ratio_range / self.segments_per_blade

            # 使用向量化生成所有截面
            segment_indices = np.arange(self.segments_per_blade)
            within_segment_indices = np.arange(num_per_segment)
            
            # 计算所有截面的比例 - 仅在每段的0.2-0.8范围内生成
            if num_per_segment > 1:
                # 映射到0.2-0.8区间，而非0-1全范围
                ratio_in_segment = 0.2 + (0.8 - 0.2) * (within_segment_indices / (num_per_segment - 1))
            else:
                # 单截面时取中间点0.5
                ratio_in_segment = np.array([0.5])
                
            # 生成网格并计算所有比例值
            seg_mesh, ratio_mesh = np.meshgrid(segment_indices, ratio_in_segment)
            # 计算最终比例：段起始比例 + 段内相对比例 * 段长度比例
            ratios = start_ratio + seg_mesh.flatten() * segment_ratio_length + ratio_mesh.flatten() * segment_ratio_length
            
            # 计算所有截面的位置
            vector_from_centroid = blade_root - self.centroid + np.outer(ratios, span_unit * span_length)
            projection_lengths = np.sum(vector_from_centroid * span_unit, axis=1)
            positions = self.centroid + projection_lengths[:, np.newaxis] * span_unit
            
            # 创建截面字典列表
            sections = []
            for idx, ratio in enumerate(ratios):
                segment = int(seg_mesh.flatten()[idx])
                sections.append({
                    'position': positions[idx],
                    'ratio': ratio,
                    'span_length': span_length * ratio,
                    'span_direction': span_unit,
                    'segment': segment
                })

            self.all_sections.append(sections)


    def _calculate_variance(self, points, edges):
        """计算截面相邻点距离的方差"""
        if len(points) < 2 or len(edges) == 0:
            return float('inf')  # 点或边不足，返回无穷大
        
        if edges.ndim == 2 and edges.shape[1] >= 2:
            # 提取所有有效边（p1 != p2）
            valid_edges = edges[edges[:, 0] != edges[:, 1]]
            if len(valid_edges) == 0:
                return float('inf')
                
            # 获取唯一的点对
            sorted_pairs = np.sort(valid_edges[:, :2], axis=1)
            unique_pairs = np.unique(sorted_pairs, axis=0)
            
            # 计算所有点对的距离
            p1_coords = points[unique_pairs[:, 0]]
            p2_coords = points[unique_pairs[:, 1]]
            distances = np.linalg.norm(p1_coords - p2_coords, axis=1)
            
            if len(distances) < 1:
                return float('inf')
                
            return np.var(distances)
        
        return float('inf')

    def _calculate_angle(self, A, B, C):
        """计算三点形成的夹角（B点的两侧连线夹角）"""
        v1 = B - A
        v2 = C - B
        dot = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        cos_theta = np.clip(dot / (norm_v1 * norm_v2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def _order_points_by_edges(self, points, edges):
        """根据边的连接关系对截面点进行排序，仅用于筛选阶段"""
        if len(points) < 3 or len(edges) == 0:
            return []
        
        # 构建邻接表：每个点的邻居索引
        adjacency = defaultdict(list)
        for edge in edges:
            u, v, _ = edge
            adjacency[u].append(v)
            adjacency[v].append(u)
        
        # 验证多边形的有效性（每个点应恰好有2个邻居）
        for idx in adjacency:
            if len(adjacency[idx]) != 2:
                return []  # 非简单多边形，返回空列表
        
        # 从第一个点开始遍历排序
        ordered_indices = []
        current_idx = 0
        prev_idx = None
        
        while len(ordered_indices) < len(points):
            ordered_indices.append(current_idx)
            # 找到下一个点（排除前一个点）
            neighbors = adjacency[current_idx]
            next_idx = neighbors[0] if neighbors[0] != prev_idx else neighbors[1]
            prev_idx, current_idx = current_idx, next_idx
            
            # 防止死循环（如果点集有问题）
            if current_idx in ordered_indices and len(ordered_indices) < len(points):
                return []
        
        return ordered_indices

    def _count_large_angles(self, points):
        """统计截面中两侧连线夹角大于40度的点的数量，仅用于筛选阶段"""
        if len(points) < 3:
            return 0
        
        count = 0
        n = len(points)
        for i in range(n):
            # 获取当前点的前一个和后一个点（形成闭合多边形）
            prev_point = points[i - 1]
            current_point = points[i]
            next_point = points[(i + 1) % n]
            
            # 计算夹角
            angle = self._calculate_angle(prev_point, current_point, next_point)
            if angle > 40:
                count += 1
        return count

    def _calculate_section_points(self, plane_origin, plane_span_direction, blade_mesh):
        """计算三维模型与平面的相交截面"""
        span_unit = plane_span_direction / np.linalg.norm(plane_span_direction)
        
        # 计算所有顶点到平面的距离
        vertices = blade_mesh.reshape(-1, 3)
        dists = np.dot(vertices - plane_origin, span_unit)
        
        # 重塑为三角形-顶点结构
        tri_dists = dists.reshape(-1, 3)
        above = tri_dists > self.epsilon
        below_or_on = ~above
        
        # 预计算所有可能的交点
        intersecting_edges = []
        
        # 处理所有三角形
        for tri_idx in range(len(blade_mesh)):
            v1, v2, v3 = blade_mesh[tri_idx]
            d1, d2, d3 = tri_dists[tri_idx]
            a1, a2, a3 = above[tri_idx]
            
            # 三个点都在平面上
            if abs(d1) < self.epsilon and abs(d2) < self.epsilon and abs(d3) < self.epsilon:
                intersecting_edges.extend([(v1, v2, tri_idx), (v2, v3, tri_idx), (v3, v1, tri_idx)])
                continue
            
            # 两个点在平面上方，一个在下方
            if a1 and a2 and not a3:
                t1 = d1 / (d1 - d3) if abs(d1 - d3) > self.epsilon else 0.5
                t2 = d2 / (d2 - d3) if abs(d2 - d3) > self.epsilon else 0.5
                p1 = v1 + t1 * (v3 - v1)
                p2 = v2 + t2 * (v3 - v2)
                intersecting_edges.append((p1, p2, tri_idx))
            elif a1 and a3 and not a2:
                t1 = d1 / (d1 - d2) if abs(d1 - d2) > self.epsilon else 0.5
                t2 = d3 / (d3 - d2) if abs(d3 - d2) > self.epsilon else 0.5
                p1 = v1 + t1 * (v2 - v1)
                p2 = v3 + t2 * (v2 - v3)
                intersecting_edges.append((p1, p2, tri_idx))
            elif a2 and a3 and not a1:
                t1 = d2 / (d2 - d1) if abs(d2 - d1) > self.epsilon else 0.5
                t2 = d3 / (d3 - d1) if abs(d3 - d1) > self.epsilon else 0.5
                p1 = v2 + t1 * (v1 - v2)
                p2 = v3 + t2 * (v1 - v3)
                intersecting_edges.append((p1, p2, tri_idx))
            
            # 两个点在平面下方，一个在上方
            elif not a1 and not a2 and a3:
                t1 = d3 / (d3 - d1) if abs(d3 - d1) > self.epsilon else 0.5
                t2 = d3 / (d3 - d2) if abs(d3 - d2) > self.epsilon else 0.5
                p1 = v3 + t1 * (v1 - v3)
                p2 = v3 + t2 * (v2 - v3)
                intersecting_edges.append((p1, p2, tri_idx))
            elif not a1 and not a3 and a2:
                t1 = d2 / (d2 - d1) if abs(d2 - d1) > self.epsilon else 0.5
                t2 = d2 / (d2 - d3) if abs(d2 - d3) > self.epsilon else 0.5
                p1 = v2 + t1 * (v1 - v2)
                p2 = v2 + t2 * (v3 - v2)
                intersecting_edges.append((p1, p2, tri_idx))
            elif not a2 and not a3 and a1:
                t1 = d1 / (d1 - d2) if abs(d1 - d2) > self.epsilon else 0.5
                t2 = d1 / (d1 - d3) if abs(d1 - d3) > self.epsilon else 0.5
                p1 = v1 + t1 * (v2 - v1)
                p2 = v1 + t2 * (v3 - v1)
                intersecting_edges.append((p1, p2, tri_idx))
            
            # 一个点在平面上，另外两个在不同侧
            elif abs(d1) < self.epsilon and a2 != a3:
                t = d2 / (d2 - d3) if abs(d2 - d3) > self.epsilon else 0.5
                p = v2 + t * (v3 - v2)
                intersecting_edges.append((v1, p, tri_idx))
            elif abs(d2) < self.epsilon and a1 != a3:
                t = d1 / (d1 - d3) if abs(d1 - d3) > self.epsilon else 0.5
                p = v1 + t * (v3 - v1)
                intersecting_edges.append((v2, p, tri_idx))
            elif abs(d3) < self.epsilon and a1 != a2:
                t = d1 / (d1 - d2) if abs(d1 - d2) > self.epsilon else 0.5
                p = v1 + t * (v2 - v1)
                intersecting_edges.append((v3, p, tri_idx))

        # 提取交点并去重
        points = []
        edges = []
        point_map = {}

        for edge in intersecting_edges:
            p1, p2, face_idx = edge
            key1 = tuple(np.round(p1, 6))
            key2 = tuple(np.round(p2, 6))

            if key1 not in point_map:
                point_map[key1] = len(points)
                points.append(p1)
            if key2 not in point_map:
                point_map[key2] = len(points)
                points.append(p2)
            edges.append((point_map[key1], point_map[key2], face_idx))

        # 过滤邻点数小于2的点及其关联边
        if len(points) == 0:
            return np.array(points), np.array(edges)

        edges_np = np.array(edges)
        if len(edges_np) == 0:
            return np.array(points), np.array(edges)
            
        # 统计每个点的邻点数量
        all_indices = np.concatenate([edges_np[:, 0], edges_np[:, 1]])
        max_index = all_indices.max() if len(all_indices) > 0 else 0
        minlength = max_index + 1
        
        point_counts = np.bincount(edges_np[:, 0], minlength=minlength)
        point_counts += np.bincount(edges_np[:, 1], minlength=minlength)
        
        # 确定需要保留的点
        to_keep = point_counts >= 2
        if not np.any(to_keep):
            return np.array([]), np.array([])
            
        # 构建旧索引到新索引的映射
        old_indices = np.where(to_keep)[0]
        old_to_new = {old: new for new, old in enumerate(old_indices)}
        
        # 过滤点和边
        filtered_points = np.array(points)[old_indices]
        valid_edges_mask = to_keep[edges_np[:, 0]] & to_keep[edges_np[:, 1]]
        valid_edges = edges_np[valid_edges_mask]
        
        filtered_edges = []
        for edge in valid_edges:
            filtered_edges.append((
                old_to_new[edge[0]],
                old_to_new[edge[1]],
                edge[2]
            ))

        return filtered_points, np.array(filtered_edges)

    def filter_sections_by_variance(self):
        """筛选出每段中符合条件的截面：
        - 排序仅用于本筛选阶段的评分计算
        - 筛选后不保留排序相关信息
        """
        if not self.all_sections or self.segments_per_blade is None:
            raise ValueError("请先完成截面生成")

        self.blade_sections = []  # 存储筛选后的截面
        start_time = time.time()
        
        for blade_idx in range(len(self.all_sections)):
            blade = self.all_sections[blade_idx]
            blade_idx_mesh = blade_idx % len(self.mesh)
            blade_mesh = np.ascontiguousarray(self.mesh[blade_idx_mesh].vectors)
            
            # 为每个截面计算筛选分数（夹角>40度的点数×方差）
            sections_with_score = []
            for section in blade:
                # 计算截面的点和边
                section_points, section_edge = self._calculate_section_points(
                    section['position'], section['span_direction'], blade_mesh
                )
                
                # 计算方差
                if len(section_points) < 3:
                    variance = float('inf')
                    count = 0
                    score = float('inf')
                else:
                    variance = self._calculate_variance(section_points, section_edge)
                    # 排序仅用于计算筛选分数，不保存排序结果
                    ordered_indices = self._order_points_by_edges(section_points, section_edge)
                    if not ordered_indices or len(ordered_indices) != len(section_points):
                        # 无法形成有效多边形
                        count = 0
                        score = float('inf')
                    else:
                        # 按顺序提取点计算大角度数量（仅用于筛选）
                        ordered_points = section_points[ordered_indices]
                        count = self._count_large_angles(ordered_points)
                        # 计算筛选分数
                        score = count * variance**2 if variance != float('inf') else float('inf')
                
                # 存储截面信息（不含排序相关数据）
                sections_with_score.append({
                    **section,
                    'variance': variance,
                    'large_angle_count': count,
                    'score': score,
                    'points': section_points,
                    'edge': section_edge
                })
            
            # 按段分组
            segment_groups = {seg: [] for seg in range(self.segments_per_blade)}
            for section in sections_with_score:
                segment_groups[section['segment']].append(section)
            
            # 验证每段是否有足够的截面
            for seg in range(self.segments_per_blade):
                if len(segment_groups[seg]) < 1:
                    raise ValueError(f"叶片 {blade_idx+1} 段 {seg+1} 缺少截面数据")

            # 每段根据是否存在无法排序的截面选择最佳截面
            blade_filtered = []
            for seg in sorted(segment_groups.keys()):
                section_group = segment_groups[seg]
                # 检查该段是否存在无法排序的截面
                has_unorderable = any(
                    len(sec['points']) < 3 or 
                    not self._order_points_by_edges(sec['points'], sec['edge'])  # 临时排序用于判断
                    for sec in section_group
                )
                
                if has_unorderable:
                    # 存在无法排序的截面，使用方差筛选
                    selected = min(section_group, key=lambda x: x['variance'])
                else:
                    # 所有截面均可排序，使用原始分数筛选
                    selected = min(section_group, key=lambda x: x['score'])
                
                blade_filtered.append(selected)
            
            self.blade_sections.append(blade_filtered)
        
        print(f"共保留 {len(self.blade_sections)} 个叶片，每叶片 {len(self.blade_sections[0])} 个截面")

    def _find_farthest_point_pair(self, points):
        """寻找点集中的最远点对（不依赖点的排序）"""
        max_dist = 0.0
        idx1, idx2 = 0, 0
        n = len(points)
        
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(points[i] - points[j])
                if dist > max_dist:
                    max_dist = dist
                    idx1, idx2 = i, j
        
        return max_dist, idx1, idx2

    def calculate_section_chords(self):
        """仅对筛选出的截面计算弦长、弦角等特征（不使用任何排序相关信息）"""
        if not self.blade_sections:
            raise ValueError("请先筛选截面")
        
        for blade_idx in range(len(self.blade_sections)):
            blade = self.blade_sections[blade_idx]
            span_unit = self.span_units[blade_idx]
            
            # 预计算切向方向
            tangential_direction = np.cross(self.axial_unit, span_unit)
            tangential_direction /= np.linalg.norm(tangential_direction)
            if self.rotor_speed_is_negative:
                tangential_direction = -tangential_direction
            
            # 对每个筛选后的截面计算完整特征（不使用排序信息）
            for section in blade:
                section_points = section['points']
                section_edge = section['edge']

                if len(section_points) < 3:
                    print(f"警告：叶片 {blade_idx+1} 截面的点太少，弦长设为0")
                    section.update({
                        'chord_length': 0,
                        'chord_angle': 0,
                        'tangential_direction': tangential_direction,
                        'chord_vector': np.zeros(3),
                        'leading_edge': np.zeros(3),
                        'leading_idx': -1,
                        'trailing_edge': np.zeros(3),
                        'trailing_idx': -1,
                        'position': section['position'],
                        'AR': 0
                    })
                    continue

                # 直接使用原始点集计算最远点对（不依赖排序）
                chord_length, idx1, idx2 = self._find_farthest_point_pair(section_points)
                chord_vector = section_points[idx1] - section_points[idx2]
                
                # 调整弦线方向与切向一致
                if np.dot(chord_vector, tangential_direction) < 0:
                    chord_vector = -chord_vector
                    idx1, idx2 = idx2, idx1
                chord_vector /= np.linalg.norm(chord_vector) if chord_length > 0 else 1

                leading_edge = section_points[idx1]
                trailing_edge = section_points[idx2]
                quarter_point = leading_edge - 0.25 * chord_length * chord_vector

                # 计算弦角
                chord_angle = self._calculate_chord_angle(
                    chord_vector, span_unit, tangential_direction
                )
                ar = self.blade_spans[blade_idx]/chord_length if chord_length != 0 else 0

                # 更新截面信息（使用原始点索引，不依赖排序）
                section.update({
                    'chord_length': chord_length,
                    'chord_angle': chord_angle,
                    'tangential_direction': tangential_direction,
                    'chord_vector': chord_vector,
                    'leading_edge': leading_edge,
                    'leading_idx': idx1,  # 原始点集中的索引
                    'trailing_edge': trailing_edge,
                    'trailing_idx': idx2,  # 原始点集中的索引
                    'position': quarter_point,
                    'AR': ar
                })
        
        return self.blade_sections

    def _calculate_chord_angle(self, chord_vector, plane_span_direction, tangential_direction):
        """计算弦线与切向方向的夹角"""
        normal = plane_span_direction / np.linalg.norm(plane_span_direction)
        projected_tangential = tangential_direction - np.dot(tangential_direction, normal) * normal
        tangential_norm = np.linalg.norm(projected_tangential)
        
        if tangential_norm < self.epsilon:
            return 0.0
            
        projected_tangential /= tangential_norm
        
        cross_vector = np.cross(projected_tangential, chord_vector)
        cross_dot_normal = np.dot(cross_vector, normal)
        dot_product = np.dot(chord_vector, projected_tangential)
        angle_rad = np.arctan2(cross_dot_normal, dot_product)
        
        # 调整角度范围
        if angle_rad > np.pi / 2:
            angle_rad -= np.pi
        elif angle_rad < -np.pi / 2:
            angle_rad += np.pi

        return angle_rad
    
def run_geometry_analysis(param_manager: ParameterManager) -> GeometryAnalyzer:
    """执行完整的叶片几何分析流程"""
    geometry = GeometryAnalyzer(param_manager)
    geometry.initialize_from_param_manager()
    geometry.analyze_blade_geometry()
    geometry.generate_section_plans()      # 1. 生成所有候选截面
    geometry.filter_sections_by_variance() # 2. 筛选出最佳截面（排序仅用于此阶段）
    geometry.calculate_section_chords()    # 3. 计算特征（不使用排序信息）
    
    return geometry