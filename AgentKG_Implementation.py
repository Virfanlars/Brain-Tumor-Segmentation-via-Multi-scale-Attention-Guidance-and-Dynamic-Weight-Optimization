"""
AgentKG: 多智能體協作的動態知識圖譜構建框架
核心實現代碼

本模塊實現了創新的跨文檔關係抽取框架，包括：
1. 多智能體協作系統
2. 動態知識圖譜管理
3. 推理鏈追踪與驗證
4. 自適應檢索機制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Any, Set
import networkx as nx
import numpy as np
from datetime import datetime
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict, deque
import asyncio
import concurrent.futures
from sentence_transformers import SentenceTransformer
import faiss

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# 核心數據結構
# ================================

@dataclass
class Entity:
    """實體數據結構"""
    id: str
    name: str
    type: str
    mentions: List[Dict]  # 包含位置、上下文等信息
    confidence: float = 0.0
    
@dataclass
class Relation:
    """關係數據結構"""
    id: str
    head: Entity
    tail: Entity
    relation_type: str
    confidence: float
    evidence: List[Dict]
    source_documents: List[str]
    timestamp: datetime
    
@dataclass
class ReasoningStep:
    """推理步驟數據結構"""
    premise: str
    inference_rule: str
    conclusion: str
    evidence: List[Dict]
    confidence: float

@dataclass
class ReasoningChain:
    """推理鏈數據結構"""
    id: str
    steps: List[ReasoningStep]
    conclusion: Relation
    overall_confidence: float
    evidence_documents: Set[str]

# ================================
# 基礎智能體抽象類
# ================================

class BaseAgent(ABC):
    """基礎智能體抽象類"""
    
    def __init__(self, agent_id: str, config: Dict):
        self.agent_id = agent_id
        self.config = config
        self.memory = {}
        self.performance_history = []
        
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """處理輸入數據的抽象方法"""
        pass
    
    def update_memory(self, key: str, value: Any):
        """更新智能體記憶"""
        self.memory[key] = value
    
    def get_performance_metrics(self) -> Dict:
        """獲取性能指標"""
        return {
            'agent_id': self.agent_id,
            'total_tasks': len(self.performance_history),
            'average_confidence': np.mean([h['confidence'] for h in self.performance_history]) if self.performance_history else 0.0
        }

# ================================
# 檢索智能體
# ================================

class RetrievalAgent(BaseAgent):
    """檢索智能體：實現自適應文檔檢索"""
    
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, config)
        
        # 初始化檢索組件
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_index = None
        self.document_embeddings = None
        
        # 檢索策略
        self.retrieval_strategies = {
            'semantic': self._semantic_retrieval,
            'temporal': self._temporal_retrieval,
            'entity_centric': self._entity_centric_retrieval,
            'relation_aware': self._relation_aware_retrieval
        }
        
        # 自適應策略選擇器
        self.strategy_weights = {strategy: 1.0 for strategy in self.retrieval_strategies}
        
    def build_document_index(self, documents: List[Dict]):
        """構建文檔索引"""
        logger.info(f"檢索智能體 {self.agent_id} 開始構建文檔索引...")
        
        # 提取文檔文本
        doc_texts = [doc['content'] for doc in documents]
        
        # 計算文檔嵌入
        self.document_embeddings = self.sentence_encoder.encode(doc_texts)
        
        # 構建FAISS索引
        dimension = self.document_embeddings.shape[1]
        self.document_index = faiss.IndexFlatIP(dimension)
        self.document_index.add(self.document_embeddings.astype(np.float32))
        
        logger.info(f"文檔索引構建完成，共 {len(documents)} 個文檔")
        
    async def process(self, query_data: Dict) -> List[Dict]:
        """處理檢索請求"""
        query = query_data['query']
        context = query_data.get('context', {})
        
        # 分析查詢意圖
        intent = self._analyze_intent(query, context)
        
        # 選擇檢索策略
        selected_strategies = self._select_strategies(intent)
        
        # 執行多策略檢索
        retrieval_results = {}
        for strategy_name in selected_strategies:
            strategy_func = self.retrieval_strategies[strategy_name]
            results = await strategy_func(query, context)
            retrieval_results[strategy_name] = results
        
        # 融合檢索結果
        final_results = self._fuse_retrieval_results(retrieval_results)
        
        # 更新性能記錄
        self.performance_history.append({
            'query': query,
            'strategies_used': selected_strategies,
            'results_count': len(final_results),
            'confidence': np.mean([r['relevance_score'] for r in final_results]) if final_results else 0.0,
            'timestamp': datetime.now()
        })
        
        return final_results
    
    def _analyze_intent(self, query: str, context: Dict) -> Dict:
        """分析查詢意圖"""
        intent = {
            'query_type': 'general',
            'entities_mentioned': [],
            'temporal_keywords': [],
            'relation_keywords': []
        }
        
        # 簡化的意圖分析邏輯
        if any(keyword in query.lower() for keyword in ['when', 'time', 'date', 'year']):
            intent['query_type'] = 'temporal'
        elif any(keyword in query.lower() for keyword in ['who', 'person', 'organization']):
            intent['query_type'] = 'entity_focused'
        elif any(keyword in query.lower() for keyword in ['relationship', 'relation', 'connection']):
            intent['query_type'] = 'relation_focused'
            
        return intent
    
    def _select_strategies(self, intent: Dict) -> List[str]:
        """基於意圖選擇檢索策略"""
        strategy_scores = {}
        
        for strategy, weight in self.strategy_weights.items():
            # 基於意圖類型調整策略得分
            if intent['query_type'] == 'temporal' and strategy == 'temporal':
                strategy_scores[strategy] = weight * 1.5
            elif intent['query_type'] == 'entity_focused' and strategy == 'entity_centric':
                strategy_scores[strategy] = weight * 1.5
            elif intent['query_type'] == 'relation_focused' and strategy == 'relation_aware':
                strategy_scores[strategy] = weight * 1.5
            else:
                strategy_scores[strategy] = weight
        
        # 選擇得分最高的策略
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        return [strategy for strategy, score in sorted_strategies[:2]]  # 選擇前2個策略
    
    async def _semantic_retrieval(self, query: str, context: Dict) -> List[Dict]:
        """語義檢索"""
        if self.document_index is None:
            return []
        
        # 計算查詢嵌入
        query_embedding = self.sentence_encoder.encode([query])
        
        # 檢索相似文檔
        scores, indices = self.document_index.search(query_embedding.astype(np.float32), k=10)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # 有效索引
                results.append({
                    'document_id': idx,
                    'relevance_score': float(score),
                    'retrieval_method': 'semantic'
                })
        
        return results
    
    async def _temporal_retrieval(self, query: str, context: Dict) -> List[Dict]:
        """時間相關檢索"""
        # 簡化實現：基於時間關鍵詞的檢索
        temporal_keywords = ['date', 'time', 'year', 'month', 'when', 'before', 'after']
        if not any(keyword in query.lower() for keyword in temporal_keywords):
            return []
        
        # 實際實現中會分析時間表達式並進行時間感知檢索
        return await self._semantic_retrieval(query, context)
    
    async def _entity_centric_retrieval(self, query: str, context: Dict) -> List[Dict]:
        """實體中心檢索"""
        # 簡化實現：基於命名實體的檢索
        # 實際實現中會進行NER並基於實體檢索
        return await self._semantic_retrieval(query, context)
    
    async def _relation_aware_retrieval(self, query: str, context: Dict) -> List[Dict]:
        """關係感知檢索"""
        # 簡化實現：基於關係關鍵詞的檢索
        relation_keywords = ['relationship', 'connection', 'link', 'associated', 'related']
        if not any(keyword in query.lower() for keyword in relation_keywords):
            return []
        
        return await self._semantic_retrieval(query, context)
    
    def _fuse_retrieval_results(self, retrieval_results: Dict[str, List[Dict]]) -> List[Dict]:
        """融合多策略檢索結果"""
        # 收集所有結果
        all_results = {}
        
        for strategy, results in retrieval_results.items():
            strategy_weight = self.strategy_weights[strategy]
            
            for result in results:
                doc_id = result['document_id']
                score = result['relevance_score'] * strategy_weight
                
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        'document_id': doc_id,
                        'relevance_score': score,
                        'contributing_strategies': [strategy]
                    }
                else:
                    all_results[doc_id]['relevance_score'] += score
                    all_results[doc_id]['contributing_strategies'].append(strategy)
        
        # 歸一化得分並排序
        final_results = list(all_results.values())
        max_score = max([r['relevance_score'] for r in final_results]) if final_results else 1.0
        
        for result in final_results:
            result['relevance_score'] /= max_score
        
        return sorted(final_results, key=lambda x: x['relevance_score'], reverse=True)

# ================================
# 抽取智能體
# ================================

class ExtractionAgent(BaseAgent):
    """抽取智能體：實現實體和關係抽取"""
    
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, config)
        
        # 加載預訓練模型
        self.model_name = config.get('model_name', 'bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # 實體解析器
        self.entity_resolver = CrossDocEntityResolver()
        
        # 關係分類器
        self.relation_classifier = nn.Sequential(
            nn.Linear(768 * 3, 512),  # 頭實體 + 尾實體 + 上下文
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, config.get('num_relations', 100))
        )
        
    async def process(self, input_data: Dict) -> List[Relation]:
        """處理關係抽取請求"""
        documents = input_data['documents']
        
        # 實體識別與消解
        entities = await self._extract_and_resolve_entities(documents)
        
        # 關係抽取
        relations = await self._extract_relations(documents, entities)
        
        # 信心度評估
        for relation in relations:
            relation.confidence = self._estimate_confidence(relation)
        
        # 更新性能記錄
        self.performance_history.append({
            'documents_processed': len(documents),
            'entities_found': len(entities),
            'relations_extracted': len(relations),
            'average_confidence': np.mean([r.confidence for r in relations]) if relations else 0.0,
            'timestamp': datetime.now()
        })
        
        return relations
    
    async def _extract_and_resolve_entities(self, documents: List[Dict]) -> List[Entity]:
        """提取並消解實體"""
        all_entities = []
        
        for doc in documents:
            # 簡化的實體抽取（實際實現中會使用NER模型）
            doc_entities = self._extract_entities_from_document(doc)
            all_entities.extend(doc_entities)
        
        # 跨文檔實體消解
        resolved_entities = self.entity_resolver.resolve(all_entities)
        
        return resolved_entities
    
    def _extract_entities_from_document(self, document: Dict) -> List[Entity]:
        """從單個文檔中抽取實體"""
        # 簡化實現：基於關鍵詞的實體識別
        content = document['content']
        entities = []
        
        # 實際實現中會使用更複雜的NER模型
        # 這裡只是示例
        import re
        
        # 尋找大寫字母開頭的短語作為實體
        patterns = [
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', 'PERSON'),
            (r'\b[A-Z][a-z]+ University\b', 'ORG'),
            (r'\b[A-Z][a-z]+ Inc\.\b', 'ORG'),
        ]
        
        for pattern, entity_type in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                entity = Entity(
                    id=f"{document['id']}_{match.start()}_{match.end()}",
                    name=match.group(),
                    type=entity_type,
                    mentions=[{
                        'document_id': document['id'],
                        'start': match.start(),
                        'end': match.end(),
                        'context': content[max(0, match.start()-50):match.end()+50]
                    }]
                )
                entities.append(entity)
        
        return entities
    
    async def _extract_relations(self, documents: List[Dict], entities: List[Entity]) -> List[Relation]:
        """提取關係"""
        relations = []
        
        # 為每對實體檢查是否存在關係
        for i, head_entity in enumerate(entities):
            for j, tail_entity in enumerate(entities[i+1:], i+1):
                relation = self._classify_relation(head_entity, tail_entity, documents)
                if relation:
                    relations.append(relation)
        
        return relations
    
    def _classify_relation(self, head_entity: Entity, tail_entity: Entity, documents: List[Dict]) -> Optional[Relation]:
        """分類實體對之間的關係"""
        # 簡化實現：基於上下文的關係分類
        # 實際實現中會使用更複雜的關係分類模型
        
        # 檢查兩個實體是否在同一文檔中共現
        common_docs = set()
        for mention1 in head_entity.mentions:
            for mention2 in tail_entity.mentions:
                if mention1['document_id'] == mention2['document_id']:
                    common_docs.add(mention1['document_id'])
        
        if not common_docs:
            return None
        
        # 簡化的關係類型判斷
        relation_type = self._determine_relation_type(head_entity, tail_entity)
        
        if relation_type != 'no_relation':
            relation = Relation(
                id=f"{head_entity.id}_{tail_entity.id}",
                head=head_entity,
                tail=tail_entity,
                relation_type=relation_type,
                confidence=0.8,  # 簡化的信心度
                evidence=[],
                source_documents=list(common_docs),
                timestamp=datetime.now()
            )
            return relation
        
        return None
    
    def _determine_relation_type(self, head_entity: Entity, tail_entity: Entity) -> str:
        """確定關係類型"""
        # 簡化的關係類型判斷邏輯
        if head_entity.type == 'PERSON' and tail_entity.type == 'ORG':
            return 'works_for'
        elif head_entity.type == 'PERSON' and tail_entity.type == 'PERSON':
            return 'knows'
        elif head_entity.type == 'ORG' and tail_entity.type == 'ORG':
            return 'collaborates_with'
        else:
            return 'related_to'
    
    def _estimate_confidence(self, relation: Relation) -> float:
        """估算關係的信心度"""
        # 簡化的信心度計算
        base_confidence = 0.5
        
        # 基於證據數量調整
        evidence_bonus = min(0.3, len(relation.evidence) * 0.1)
        
        # 基於實體類型匹配度調整
        type_bonus = 0.2 if self._relation_type_matches_entities(relation) else 0.0
        
        return min(1.0, base_confidence + evidence_bonus + type_bonus)
    
    def _relation_type_matches_entities(self, relation: Relation) -> bool:
        """檢查關係類型是否與實體類型匹配"""
        # 簡化的匹配邏輯
        expected_types = {
            'works_for': [('PERSON', 'ORG')],
            'knows': [('PERSON', 'PERSON')],
            'collaborates_with': [('ORG', 'ORG')]
        }
        
        relation_type = relation.relation_type
        head_type = relation.head.type
        tail_type = relation.tail.type
        
        if relation_type in expected_types:
            return (head_type, tail_type) in expected_types[relation_type]
        
        return True  # 對於未知關係類型，假設匹配

# ================================
# 跨文檔實體消解器
# ================================

class CrossDocEntityResolver:
    """跨文檔實體消解器"""
    
    def __init__(self):
        self.similarity_threshold = 0.8
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def resolve(self, entities: List[Entity]) -> List[Entity]:
        """解析跨文檔實體"""
        if not entities:
            return []
        
        # 計算實體相似度矩陣
        similarity_matrix = self._calculate_similarity_matrix(entities)
        
        # 聚類相似實體
        entity_clusters = self._cluster_entities(similarity_matrix, entities)
        
        # 合併聚類中的實體
        resolved_entities = self._merge_clustered_entities(entity_clusters)
        
        return resolved_entities
    
    def _calculate_similarity_matrix(self, entities: List[Entity]) -> np.ndarray:
        """計算實體相似度矩陣"""
        n = len(entities)
        similarity_matrix = np.zeros((n, n))
        
        # 計算實體名稱嵌入
        entity_names = [entity.name for entity in entities]
        embeddings = self.sentence_encoder.encode(entity_names)
        
        # 計算相似度
        for i in range(n):
            for j in range(i, n):
                # 名稱相似度
                name_sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                
                # 類型匹配
                type_match = 1.0 if entities[i].type == entities[j].type else 0.0
                
                # 綜合相似度
                overall_sim = 0.7 * name_sim + 0.3 * type_match
                
                similarity_matrix[i][j] = overall_sim
                similarity_matrix[j][i] = overall_sim
        
        return similarity_matrix
    
    def _cluster_entities(self, similarity_matrix: np.ndarray, entities: List[Entity]) -> List[List[int]]:
        """聚類相似實體"""
        n = len(entities)
        visited = [False] * n
        clusters = []
        
        for i in range(n):
            if not visited[i]:
                cluster = [i]
                visited[i] = True
                
                # 找到所有相似的實體
                for j in range(i + 1, n):
                    if not visited[j] and similarity_matrix[i][j] > self.similarity_threshold:
                        cluster.append(j)
                        visited[j] = True
                
                clusters.append(cluster)
        
        return clusters
    
    def _merge_clustered_entities(self, entity_clusters: List[List[int]]) -> List[Entity]:
        """合併聚類中的實體"""
        merged_entities = []
        
        for cluster_indices in entity_clusters:
            if len(cluster_indices) == 1:
                # 單個實體，直接添加
                merged_entities.append(entities[cluster_indices[0]])
            else:
                # 多個實體，需要合併
                merged_entity = self._merge_entities([entities[i] for i in cluster_indices])
                merged_entities.append(merged_entity)
        
        return merged_entities
    
    def _merge_entities(self, entities: List[Entity]) -> Entity:
        """合併多個實體為一個"""
        # 選擇最常見的名稱
        name_counts = defaultdict(int)
        for entity in entities:
            name_counts[entity.name] += 1
        most_common_name = max(name_counts.items(), key=lambda x: x[1])[0]
        
        # 選擇最常見的類型
        type_counts = defaultdict(int)
        for entity in entities:
            type_counts[entity.type] += 1
        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
        
        # 合併所有提及
        all_mentions = []
        for entity in entities:
            all_mentions.extend(entity.mentions)
        
        # 計算平均信心度
        avg_confidence = np.mean([entity.confidence for entity in entities])
        
        # 創建合併後的實體
        merged_entity = Entity(
            id=f"merged_{entities[0].id}",
            name=most_common_name,
            type=most_common_type,
            mentions=all_mentions,
            confidence=avg_confidence
        )
        
        return merged_entity

# ================================
# 推理智能體
# ================================

class ReasoningAgent(BaseAgent):
    """推理智能體：實現多跳推理和邏輯檢查"""
    
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, config)
        self.max_reasoning_depth = config.get('max_reasoning_depth', 3)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        
    async def process(self, input_data: Dict) -> Dict:
        """處理推理請求"""
        relations = input_data['relations']
        knowledge_graph = input_data['knowledge_graph']
        
        # 構建推理鏈
        reasoning_chains = await self._build_reasoning_chains(relations, knowledge_graph)
        
        # 邏輯一致性檢查
        valid_chains = self._validate_reasoning_chains(reasoning_chains)
        
        # 推斷新關係
        inferred_relations = self._infer_new_relations(valid_chains, knowledge_graph)
        
        # 更新性能記錄
        self.performance_history.append({
            'input_relations': len(relations),
            'reasoning_chains_built': len(reasoning_chains),
            'valid_chains': len(valid_chains),
            'inferred_relations': len(inferred_relations),
            'timestamp': datetime.now()
        })
        
        return {
            'reasoning_chains': valid_chains,
            'inferred_relations': inferred_relations,
            'reasoning_statistics': self._calculate_reasoning_stats(valid_chains)
        }
    
    async def _build_reasoning_chains(self, relations: List[Relation], knowledge_graph) -> List[ReasoningChain]:
        """構建推理鏈"""
        reasoning_chains = []
        
        for relation in relations:
            # 為每個關係尋找支持的推理路徑
            chains = self._find_reasoning_paths(relation, knowledge_graph)
            reasoning_chains.extend(chains)
        
        return reasoning_chains
    
    def _find_reasoning_paths(self, target_relation: Relation, knowledge_graph) -> List[ReasoningChain]:
        """尋找支持目標關係的推理路徑"""
        chains = []
        
        # 簡化實現：基於圖搜索的推理路徑發現
        start_entity = target_relation.head.id
        end_entity = target_relation.tail.id
        
        # 使用廣度優先搜索尋找路徑
        paths = self._bfs_find_paths(start_entity, end_entity, knowledge_graph, self.max_reasoning_depth)
        
        for path in paths:
            chain = self._construct_reasoning_chain(path, target_relation, knowledge_graph)
            if chain.overall_confidence > self.confidence_threshold:
                chains.append(chain)
        
        return chains
    
    def _bfs_find_paths(self, start: str, end: str, knowledge_graph, max_depth: int) -> List[List[str]]:
        """使用BFS尋找路徑"""
        if not hasattr(knowledge_graph, 'graph'):
            return []
        
        paths = []
        queue = deque([(start, [start])])
        visited = set()
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current == end and len(path) > 1:
                paths.append(path)
                continue
            
            if current in visited:
                continue
            visited.add(current)
            
            # 獲取鄰居節點
            neighbors = knowledge_graph.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in path:  # 避免循環
                    queue.append((neighbor, path + [neighbor]))
        
        return paths
    
    def _construct_reasoning_chain(self, path: List[str], target_relation: Relation, knowledge_graph) -> ReasoningChain:
        """構建推理鏈"""
        steps = []
        evidence_docs = set()
        overall_confidence = 1.0
        
        for i in range(len(path) - 1):
            current_entity = path[i]
            next_entity = path[i + 1]
            
            # 獲取兩個實體間的關係
            relation_info = knowledge_graph.get_relation(current_entity, next_entity)
            
            if relation_info:
                step = ReasoningStep(
                    premise=f"{current_entity} {relation_info['type']} {next_entity}",
                    inference_rule=relation_info['type'],
                    conclusion=f"Connection between {current_entity} and {next_entity}",
                    evidence=relation_info.get('evidence', []),
                    confidence=relation_info.get('confidence', 0.8)
                )
                
                steps.append(step)
                evidence_docs.update(relation_info.get('source_documents', []))
                overall_confidence *= step.confidence
        
        chain = ReasoningChain(
            id=f"chain_{target_relation.id}_{len(steps)}",
            steps=steps,
            conclusion=target_relation,
            overall_confidence=overall_confidence,
            evidence_documents=evidence_docs
        )
        
        return chain
    
    def _validate_reasoning_chains(self, reasoning_chains: List[ReasoningChain]) -> List[ReasoningChain]:
        """驗證推理鏈的邏輯一致性"""
        valid_chains = []
        
        for chain in reasoning_chains:
            if self._is_logically_consistent(chain):
                valid_chains.append(chain)
        
        return valid_chains
    
    def _is_logically_consistent(self, chain: ReasoningChain) -> bool:
        """檢查推理鏈的邏輯一致性"""
        # 簡化的邏輯一致性檢查
        
        # 1. 檢查信心度閾值
        if chain.overall_confidence < self.confidence_threshold:
            return False
        
        # 2. 檢查推理步驟的連貫性
        for i, step in enumerate(chain.steps):
            if step.confidence < 0.5:  # 單步信心度太低
                return False
        
        # 3. 檢查推理鏈長度
        if len(chain.steps) > self.max_reasoning_depth:
            return False
        
        return True
    
    def _infer_new_relations(self, valid_chains: List[ReasoningChain], knowledge_graph) -> List[Relation]:
        """基於有效推理鏈推斷新關係"""
        inferred_relations = []
        
        for chain in valid_chains:
            # 基於推理鏈推斷新的關係
            if len(chain.steps) >= 2:
                # 傳遞性推理：如果 A->B 和 B->C，則可能 A->C
                first_step = chain.steps[0]
                last_step = chain.steps[-1]
                
                # 提取實體
                start_entity = self._extract_entity_from_premise(first_step.premise, 'start')
                end_entity = self._extract_entity_from_premise(last_step.premise, 'end')
                
                if start_entity and end_entity:
                    inferred_relation = Relation(
                        id=f"inferred_{start_entity}_{end_entity}",
                        head=Entity(id=start_entity, name=start_entity, type='INFERRED', mentions=[]),
                        tail=Entity(id=end_entity, name=end_entity, type='INFERRED', mentions=[]),
                        relation_type='inferred_connection',
                        confidence=chain.overall_confidence * 0.8,  # 降低推斷關係的信心度
                        evidence=[{'reasoning_chain_id': chain.id}],
                        source_documents=list(chain.evidence_documents),
                        timestamp=datetime.now()
                    )
                    
                    inferred_relations.append(inferred_relation)
        
        return inferred_relations
    
    def _extract_entity_from_premise(self, premise: str, position: str) -> Optional[str]:
        """從前提中提取實體"""
        # 簡化的實體提取邏輯
        parts = premise.split()
        if position == 'start' and len(parts) >= 1:
            return parts[0]
        elif position == 'end' and len(parts) >= 3:
            return parts[-1]
        return None
    
    def _calculate_reasoning_stats(self, valid_chains: List[ReasoningChain]) -> Dict:
        """計算推理統計信息"""
        if not valid_chains:
            return {}
        
        return {
            'total_chains': len(valid_chains),
            'average_confidence': np.mean([chain.overall_confidence for chain in valid_chains]),
            'average_chain_length': np.mean([len(chain.steps) for chain in valid_chains]),
            'total_evidence_documents': len(set().union(*[chain.evidence_documents for chain in valid_chains]))
        }

# ================================
# 主框架類
# ================================

class AgentKGFramework:
    """AgentKG主框架"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.agents = {}
        self.knowledge_graph = DynamicKnowledgeGraph()
        
        # 初始化智能體
        self._initialize_agents()
        
        # 設置智能體間通信
        self.message_queue = asyncio.Queue()
        
    def _initialize_agents(self):
        """初始化所有智能體"""
        agent_configs = self.config.get('agents', {})
        
        # 檢索智能體
        self.agents['retrieval'] = RetrievalAgent('retrieval_001', agent_configs.get('retrieval', {}))
        
        # 抽取智能體
        self.agents['extraction'] = ExtractionAgent('extraction_001', agent_configs.get('extraction', {}))
        
        # 推理智能體
        self.agents['reasoning'] = ReasoningAgent('reasoning_001', agent_configs.get('reasoning', {}))
        
        logger.info(f"已初始化 {len(self.agents)} 個智能體")
    
    async def process_documents(self, documents: List[Dict], query: str = None) -> Dict:
        """處理文檔集合，執行跨文檔關係抽取"""
        logger.info(f"開始處理 {len(documents)} 個文檔")
        
        # 1. 文檔檢索（如果提供了查詢）
        if query:
            retrieval_result = await self.agents['retrieval'].process({
                'query': query,
                'context': {'documents': documents}
            })
            relevant_docs = [documents[r['document_id']] for r in retrieval_result[:10]]
        else:
            relevant_docs = documents
        
        # 2. 關係抽取
        extraction_result = await self.agents['extraction'].process({
            'documents': relevant_docs
        })
        
        # 3. 推理鏈構建
        reasoning_result = await self.agents['reasoning'].process({
            'relations': extraction_result,
            'knowledge_graph': self.knowledge_graph
        })
        
        # 4. 更新知識圖譜
        self.knowledge_graph.update_with_relations(
            extraction_result + reasoning_result['inferred_relations']
        )
        
        # 5. 生成結果報告
        result = {
            'extracted_relations': extraction_result,
            'reasoning_chains': reasoning_result['reasoning_chains'],
            'inferred_relations': reasoning_result['inferred_relations'],
            'knowledge_graph_stats': self.knowledge_graph.get_statistics(),
            'agent_performance': {
                agent_id: agent.get_performance_metrics() 
                for agent_id, agent in self.agents.items()
            }
        }
        
        logger.info("文檔處理完成")
        return result

# ================================
# 動態知識圖譜
# ================================

class DynamicKnowledgeGraph:
    """動態知識圖譜"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.relation_history = []
        self.version = 0
        
    def update_with_relations(self, relations: List[Relation]):
        """使用新關係更新知識圖譜"""
        for relation in relations:
            self._add_relation(relation)
        self.version += 1
        
    def _add_relation(self, relation: Relation):
        """添加關係到知識圖譜"""
        head_id = relation.head.id
        tail_id = relation.tail.id
        
        # 添加節點
        self.graph.add_node(head_id, **{
            'name': relation.head.name,
            'type': relation.head.type,
            'confidence': relation.head.confidence
        })
        
        self.graph.add_node(tail_id, **{
            'name': relation.tail.name,
            'type': relation.tail.type,
            'confidence': relation.tail.confidence
        })
        
        # 添加邊
        self.graph.add_edge(head_id, tail_id, **{
            'relation_type': relation.relation_type,
            'confidence': relation.confidence,
            'evidence': relation.evidence,
            'source_documents': relation.source_documents,
            'timestamp': relation.timestamp
        })
        
        # 記錄歷史
        self.relation_history.append({
            'action': 'add',
            'relation': relation,
            'version': self.version,
            'timestamp': datetime.now()
        })
    
    def get_neighbors(self, entity_id: str) -> List[str]:
        """獲取實體的鄰居"""
        if entity_id in self.graph:
            return list(self.graph.neighbors(entity_id))
        return []
    
    def get_relation(self, head_id: str, tail_id: str) -> Optional[Dict]:
        """獲取兩個實體間的關係"""
        if self.graph.has_edge(head_id, tail_id):
            return self.graph.edges[head_id, tail_id]
        return None
    
    def get_statistics(self) -> Dict:
        """獲取知識圖譜統計信息"""
        return {
            'num_entities': self.graph.number_of_nodes(),
            'num_relations': self.graph.number_of_edges(),
            'version': self.version,
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0.0
        }

# ================================
# 使用示例
# ================================

def main():
    """主函數：演示AgentKG框架的使用"""
    
    # 配置
    config = {
        'agents': {
            'retrieval': {'model_name': 'bert-base-uncased'},
            'extraction': {'model_name': 'bert-base-uncased', 'num_relations': 50},
            'reasoning': {'max_reasoning_depth': 3, 'confidence_threshold': 0.6}
        }
    }
    
    # 初始化框架
    framework = AgentKGFramework(config)
    
    # 示例文檔
    documents = [
        {
            'id': 'doc1',
            'content': 'John Smith works at Google Inc. He is a software engineer.'
        },
        {
            'id': 'doc2', 
            'content': 'Google Inc. collaborates with Stanford University on AI research.'
        },
        {
            'id': 'doc3',
            'content': 'Stanford University has many talented researchers including John Smith.'
        }
    ]
    
    # 異步處理
    async def process():
        result = await framework.process_documents(documents, "Find relationships between people and organizations")
        
        print("=== 抽取結果 ===")
        print(f"發現 {len(result['extracted_relations'])} 個關係")
        print(f"構建 {len(result['reasoning_chains'])} 個推理鏈")
        print(f"推斷 {len(result['inferred_relations'])} 個新關係")
        
        return result
    
    # 運行示例
    if __name__ == "__main__":
        import asyncio
        result = asyncio.run(process())

if __name__ == "__main__":
    main() 