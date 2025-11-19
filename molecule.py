'''
import pickle
#import dgl
import math
import torch
import pysmiles
import deepchem as dc
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from pathlib import Path
from .base import Featurizer
from ..utils import get_logger, canonicalize

from mol2vec.features import (
    mol2alt_sentence,
    mol2sentence,
    MolSentence,
    sentences2vec,
)
from gensim.models import word2vec
#from dgl.dataloading import GraphDataLoader
#from dgl.nn import GraphConv, GATConv, SAGEConv, SGConv, TAGConv
#from dgl.nn.pytorch.glob import SumPooling
from torch.nn import ModuleList
from torch.nn.functional import one_hot

logg = get_logger()

MODEL_CACHE_DIR = Path(
    "/afs/csail.mit.edu/u/s/samsl/Work/Adapting_PLM_DTI/models"
)
'''
'''
class Mol2VecFeaturizer(Featurizer):
    def __init__(self, radius: int = 1, save_dir: Path = Path().absolute()):
        super().__init__("Mol2Vec", 300)

        self._radius = radius
        self._model = word2vec.Word2Vec.load(
            f"{MODEL_CACHE_DIR}/mol2vec_saved/model_300dim.pkl"
        )

    def _transform(self, smile: str) -> torch.Tensor:

        molecule = Chem.MolFromSmiles(smile)
        try:
            sentence = MolSentence(mol2alt_sentence(molecule, self._radius))
            wide_vector = sentences2vec(sentence, self._model, unseen="UNK")
            feats = wide_vector.mean(axis=0)
        except Exception:
            feats = np.zeros(self.shape)

        feats = torch.from_numpy(feats).squeeze().float()
        return feats


class MorganFeaturizer(Featurizer):
    def __init__(
        self,
        shape: int = 2048,
        radius: int = 2,
        save_dir: Path = Path().absolute(),
    ):
        super().__init__("Morgan", shape, save_dir)

        self._radius = radius

    def smiles_to_morgan(self, smile: str):
        """
        Convert smiles into Morgan Fingerprint.
        :param smile: SMILES string
        :type smile: str
        :return: Morgan fingerprint
        :rtype: np.ndarray
        """
        try:
            smile = canonicalize(smile)
            mol = Chem.MolFromSmiles(smile)
            features_vec = AllChem.GetMorganFingerprintAsBitVect(
                mol, self._radius, nBits=self.shape
            )
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features)
        except Exception as e:
            logg.error(
                f"rdkit not found this smiles for morgan: {smile} convert to all 0 features"
            )
            logg.error(e)
            features = np.zeros((self.shape,))
        return features

    def _transform(self, smile: str) -> torch.Tensor:
        # feats = torch.from_numpy(self._featurizer(smile)).squeeze().float()
        feats = (
            torch.from_numpy(self.smiles_to_morgan(smile)).squeeze().float()
        )
        if feats.shape[0] != self.shape:
            logg.warning("Failed to featurize: appending zero vector")
            feats = torch.zeros(self.shape)
        return feats


class GNN(torch.nn.Module):
    def __init__(self, gnn, n_layer, feature_len, dim):
        super(GNN, self).__init__()
        self.gnn = gnn
        self.n_layer = n_layer
        self.feature_len = feature_len
        self.dim = dim
        self.gnn_layers = ModuleList([])
        if gnn in ["gcn", "gat", "sage", "tag"]:
            for i in range(n_layer):
                if gnn == "gcn":
                    self.gnn_layers.append(
                        GraphConv(
                            in_feats=feature_len if i == 0 else dim,
                            out_feats=dim,
                            activation=None
                            if i == n_layer - 1
                            else torch.relu,
                        )
                    )
                elif gnn == "gat":
                    num_heads = (
                        16  # make sure that dim is dividable by num_heads
                    )
                    self.gnn_layers.append(
                        GATConv(
                            in_feats=feature_len if i == 0 else dim,
                            out_feats=dim // num_heads,
                            activation=None
                            if i == n_layer - 1
                            else torch.relu,
                            num_heads=num_heads,
                        )
                    )
                elif gnn == "sage":
                    agg = "pool"
                    self.gnn_layers.append(
                        SAGEConv(
                            in_feats=feature_len if i == 0 else dim,
                            out_feats=dim,
                            activation=None
                            if i == n_layer - 1
                            else torch.relu,
                            aggregator_type=agg,
                        )
                    )
                elif gnn == "tag":
                    hops = 2
                    self.gnn_layers.append(
                        TAGConv(
                            in_feats=feature_len if i == 0 else dim,
                            out_feats=dim,
                            activation=None
                            if i == n_layer - 1
                            else torch.relu,
                            k=hops,
                        )
                    )
        elif gnn == "sgc":
            self.gnn_layers.append(
                SGConv(in_feats=feature_len, out_feats=dim, k=n_layer)
            )
        else:
            raise ValueError("unknown GNN model")
        self.pooling_layer = SumPooling()
        self.factor = None

    def forward(self, graph):
        feature = graph.ndata["feature"]
        h = one_hot(feature, num_classes=self.feature_len)
        h = torch.sum(h, dim=1, dtype=torch.float)
        for layer in self.gnn_layers:
            h = layer(graph, h)
            if self.gnn == "gat":
                h = torch.reshape(h, [h.size()[0], -1])
        if self.factor is None:
            self.factor = math.sqrt(self.dim) / float(
                torch.mean(torch.linalg.norm(h, dim=1))
            )
        h *= self.factor
        graph_embedding = self.pooling_layer(graph, h)
        return graph_embedding


class GraphDataset(dgl.data.DGLDataset):
    def __init__(self, path_to_model, smiles_list, gpu):
        self.path = path_to_model
        self.smiles_list = smiles_list
        self.gpu = gpu
        self.parsed = []
        self.graphs = []
        super().__init__(name="graph_dataset")

    def process(self):
        with open(self.path + "/feature_enc.pkl", "rb") as f:
            feature_encoder = pickle.load(f)
        for i, smiles in enumerate(self.smiles_list):
            try:
                raw_graph = pysmiles.read_smiles(
                    smiles, zero_order_bonds=False
                )
                dgl_graph = networkx_to_dgl(raw_graph, feature_encoder)
                self.graphs.append(dgl_graph)
                self.parsed.append(i)
            except Exception as e:
                logg.error(e)
                # print('ERROR: No. %d smiles is not parsed successfully' % i)
        # print('the number of smiles successfully parsed: %d' % len(self.parsed))
        # print('the number of smiles failed to be parsed: %d' % (len(self.smiles_list) - len(self.parsed)))
        if torch.cuda.is_available() and self.gpu is not None:
            self.graphs = [
                graph.to("cuda:" + str(self.gpu)) for graph in self.graphs
            ]

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)


def networkx_to_dgl(raw_graph, feature_encoder):
    attribute_names = ["element", "charge", "aromatic", "hcount"]
    # add edges
    src = [s for (s, _) in raw_graph.edges]
    dst = [t for (_, t) in raw_graph.edges]
    graph = dgl.graph((src, dst), num_nodes=len(raw_graph.nodes))
    # add node features
    node_features = []
    for i in range(len(raw_graph.nodes)):
        raw_feature = raw_graph.nodes[i]
        numerical_feature = []
        for j in attribute_names:
            if raw_feature[j] in feature_encoder[j]:
                numerical_feature.append(feature_encoder[j][raw_feature[j]])
            else:
                numerical_feature.append(feature_encoder[j]["unknown"])
        node_features.append(numerical_feature)
    node_features = torch.tensor(node_features)
    graph.ndata["feature"] = node_features
    # transform to bi-directed graph with self-loops
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    return graph


class MolEFeaturizer(object):
    def __init__(self, path_to_model, gpu=0):
        self.path_to_model = path_to_model
        self.gpu = gpu
        with open(path_to_model + "/hparams.pkl", "rb") as f:
            hparams = pickle.load(f)
        self.mole = GNN(
            hparams["gnn"],
            hparams["layer"],
            hparams["feature_len"],
            hparams["dim"],
        )
        self.dim = hparams["dim"]
        if torch.cuda.is_available() and gpu is not None:
            self.mole.load_state_dict(torch.load(path_to_model + "/model.pt"))
            self.mole = self.mole.cuda(gpu)
        else:
            self.mole.load_state_dict(
                torch.load(
                    path_to_model + "/model.pt",
                    map_location=torch.device("cpu"),
                )
            )

    def transform(self, smiles_list, batch_size=None, data=None):
        if data is None:
            data = GraphDataset(self.path_to_model, smiles_list, self.gpu)
        dataloader = GraphDataLoader(
            data,
            batch_size=batch_size
            if batch_size is not None
            else len(smiles_list),
        )
        all_embeddings = np.zeros((len(smiles_list), self.dim), dtype=float)
        flags = np.zeros(len(smiles_list), dtype=bool)
        res = []
        with torch.no_grad():
            self.mole.eval()
            for graphs in dataloader:
                graph_embeddings = self.mole(graphs)
                res.append(graph_embeddings)
            res = torch.cat(res, dim=0).cpu().numpy()
        all_embeddings[data.parsed, :] = res
        flags[data.parsed] = True
        return all_embeddings, flags


class MolRFeaturizer(Featurizer):
    def __init__(
        self,
        shape: int = 1024,
        save_dir: Path = Path().absolute(),
    ):
        super().__init__("MolR", shape, save_dir)

        self.path_to_model = f"{MODEL_CACHE_DIR}/molr_saved/gcn_1024"
        self._molE_featurizer = MolEFeaturizer(
            path_to_model=self.path_to_model
        )

    def _transform(self, smile: str) -> torch.Tensor:
        smile = canonicalize(smile)
        try:
            embeddings, _ = self._molE_featurizer.transform([smile])
        except NotImplementedError:
            embeddings = np.zeros(self.shape)
        tens = torch.from_numpy(embeddings).squeeze().float()
        return tens
'''



# molecule.py

import torch
from pathlib import Path
from rdkit import Chem
from train_pubchem_light import LightningModule
import numpy as np
from .base import Featurizer  # Assuming Featurizer is defined in base.py
import yaml
from rdkit import Chem
import pandas as pd
from argparse import Namespace
import torch
from fast_transformers.masking import LengthMask as LM
from transformers import BertTokenizer
import regex as re
import h5py

PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class MolTranBertTokenizer(BertTokenizer):
    def __init__(self, vocab_file: str = '',
                 do_lower_case=False,
                 unk_token='<pad>',
                 sep_token='<eos>',
                 pad_token='<pad>',
                 cls_token='<bos>',
                 mask_token='<mask>',
                 **kwargs):
        super().__init__(vocab_file,
                         unk_token=unk_token,
                         sep_token=sep_token,
                         pad_token=pad_token,
                         cls_token=cls_token,
                         mask_token=mask_token,
                         **kwargs)

        self.regex_tokenizer = re.compile(PATTERN)
        self.wordpiece_tokenizer = None
        self.basic_tokenizer = None

    def _tokenize(self, text):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens

    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).strip()
        return out_string


# Load MoLFormer Model and Tokenizer
class MoLFormerFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute()):
        super().__init__("MoLFormer", 300, save_dir)  # Assuming 300 dimensions for embedding

        # Load configuration and pretrained model
        with open('/data/hanyuhang/ssw_lab/qbx/drug/SMILE/Pretrained MoLFormer/hparams.yaml', 'r') as f:
            config = Namespace(**yaml.safe_load(f))

        self.tokenizer = MolTranBertTokenizer('/data/hanyuhang/ssw_lab/qbx/drug/SMILE/Pretrained MoLFormer/bert_vocab.txt')
        ckpt = '/data/hanyuhang/ssw_lab/qbx/drug/SMILE/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
        self.model = LightningModule(config, self.tokenizer.vocab).load_from_checkpoint(ckpt, config=config,
                                                                                        vocab=self.tokenizer.vocab)

    def _transform(self, smile: str) -> torch.Tensor:
        """Generate embeddings for a single SMILES string using MoLFormer"""
        try:
            smiles = [smile]  # Convert single SMILES to list for batching
            embedding = self._embed(smiles).squeeze()  # Generate embedding
        except Exception as e:
            print(f"Error generating embedding for SMILES: {smile}, Error: {e}")
            embedding = torch.zeros(self.shape)  # Return zero vector if embedding fails
        return embedding

    def _embed(self, smiles_list, batch_size=64):
        """Embed a list of SMILES strings"""
        self.model.eval()
        embeddings = []
        for batch in self._batch_split(smiles_list, batch_size=batch_size):
            batch_enc = self.tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
            idx, mask = torch.tensor(batch_enc['input_ids']), torch.tensor(batch_enc['attention_mask'])
            with torch.no_grad():
                token_embeddings = self.model.blocks(self.model.tok_emb(idx), length_mask=LM(mask.sum(-1)))

            # Average pooling over tokens
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            embeddings.append(embedding.detach().cpu())

        return torch.cat(embeddings)

    def _batch_split(self, data, batch_size=64):
        i = 0
        while i < len(data):
            yield data[i:min(i + batch_size, len(data))]
            i += batch_size


# Example usage
def canonicalize(smile):
    """Normalize SMILES using RDKit"""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smile), canonical=True, isomericSmiles=False)


def generate_embeddings_for_file(input_csv, output_h5):
    # 读取CSV文件并规范化SMILES
    df = pd.read_csv(input_csv)
    df['SMILES'] = df['SMILES'].apply(canonicalize)

    # 实例化 MoLFormerFeaturizer
    featurizer = MoLFormerFeaturizer()

    # 生成嵌入
    smiles_list = df['SMILES'].tolist()
    embeddings = featurizer._embed(smiles_list).numpy()

    # 保存嵌入结果为h5文件
    with h5py.File(output_h5, 'w') as hf:
        hf.create_dataset('embeddings', data=embeddings)
        hf.create_dataset('smiles', data=np.array(smiles_list, dtype='S'))  # 将SMILES存储为字节字符串



# 5. 为训练集、验证集和测试集生成嵌入并保存
train_csv = '/data/hanyuhang/ssw_lab/qbx/drug/SMILE/dataset/DAVIS/train.csv'
test_csv = '/data/hanyuhang/ssw_lab/qbx/drug/SMILE/dataset/DAVIS/test.csv'
val_csv = '/data/hanyuhang/ssw_lab/qbx/drug/SMILE/dataset/DAVIS/val.csv'

generate_embeddings_for_file(train_csv, 'train_embeddings.h5')
generate_embeddings_for_file(test_csv, 'test_embeddings.h5')
generate_embeddings_for_file(val_csv, 'val_embeddings.h5')
