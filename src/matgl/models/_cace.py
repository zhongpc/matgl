"""Implementation of CACE model.

A Cartesian based equivariant GNN model. For more details on CACE,
please refer to::


"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional, Any, Dict, List

import dgl
import torch
from torch import nn

import matgl
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
)

from matgl.layers._readout import MLPReadOut, WeightedReadOut, WeightedReadOut_norepeat

if TYPE_CHECKING:
    from matgl.graph.converters import GraphConverter

logger = logging.getLogger(__file__)

from matgl.cace.modules.blocks import build_mlp
from matgl.cace.modules import (
    NodeEncoder, 
    NodeEmbedding, 
    EdgeEncoder,
    AngularComponent, 
    AngularComponent_GPU,
    SharedRadialLinearTransform,
    Symmetrizer,
    #Symmetrizer_JIT,
    MessageAr, 
    MessageBchi,
    NodeMemory,
    BesselRBF,
    PolynomialCutoff,
    )
from matgl.cace.modules import (
    get_edge_vectors_and_lengths,
    ) 

from matgl.cace.tools import elementwise_multiply_3tensors, scatter_sum


from pymatgen.core.periodic_table import Element
DEFAULT_ELEMENTS = tuple(el.symbol for el in Element if el.Z < 95)
DEFAULT_ATOMIC_NUMBERS = tuple(el.Z for el in Element if el.Z < 95)


class CACE_LR(nn.Module):

    def __init__(
        self,
        zs: List[int] = list(DEFAULT_ATOMIC_NUMBERS),
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        n_atom_basis: int = 3,
        n_rbf: int = 6,
        trainable_rbf: bool = True,
        cutoff_fn_p: int = 6,
        cutoff: float = 5.5,
        max_l: int = 3,
        max_nu: int = 3,
        num_message_passing: int = 1,
        node_encoder: Optional[nn.Module] = None,
        edge_encoder: Optional[nn.Module] = None,
        type_message_passing: List[str] = ["M", "Ar", "Bchi"],
        args_message_passing: Dict[str, Any] = {"M": {}, "Ar": {}, "Bchi": {}},
        embed_receiver_nodes: bool = False,
        atom_embedding_random_seed: List[int] = [42, 42], 
        n_radial_basis: Optional[int] = None,
        avg_num_neighbors: float = 10.0,
        # device: torch.device = torch.device("cuda"),
        timeit: bool = False,
        keep_node_features_A: bool = False,
        forward_features: List[str] = [],
        num_targets: int = 1,
        readout_type: Literal["weighted_mlp", "mlp"] = "mlp",
        atomwise_hidden: List[int] = [64, 32],
        latent_charge_hidden: List[int] = [64, 32],
    ):
        """
        Args:
            zs: list of atomic numbers
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            edge_coding: layer for encoding edge type
            cutoff: cutoff radius
            radial_basis: layer for expanding interatomic distances in a basis set
            n_radial_basis: number of radial embedding dimensions
            cutoff_fn: cutoff function
            cutoff: cutoff radius
            max_l: the maximum l considered in the angular basis
            max_nu: the maximum correlation order
            num_message_passing: number of message passing layers
            avg_num_neighbors: average number of neighbors per atom, used for normalization
        """
        super().__init__()
        self.element_types = element_types  # type: ignore

        self.zs = zs # list(DEFAULT_ATOMIC_NUMBERS)
        self.nz = len(self.zs) # number of elements

        self.n_atom_basis = n_atom_basis
        self.n_rbf = n_rbf
        self.trainable_rbf = trainable_rbf
        self.cutoff_fn_p = cutoff_fn_p
        self.cutoff = cutoff
        self.max_l = max_l
        self.max_nu = max_nu
        self.mp_norm_factor = 1.0/(avg_num_neighbors)**0.5 # normalization factor for message passing
        self.keep_node_features_A = keep_node_features_A
        self.forward_features = forward_features
        self.readout_type = readout_type

        # layers
        if node_encoder is None:
            self.node_onehot = NodeEncoder(self.zs)
            self.nz = len(self.zs) # number of elements
        else:
            self.node_onehot = node_encoder
            self.nz = node_encoder.embedding_dim

        # sender node embedding
        self.node_embedding_sender = NodeEmbedding(
                         node_dim=self.nz, embedding_dim=self.n_atom_basis, random_seed=atom_embedding_random_seed[0]
                         )
        if embed_receiver_nodes:
            self.node_embedding_receiver = NodeEmbedding(
                         node_dim=self.nz, embedding_dim=self.n_atom_basis, random_seed=atom_embedding_random_seed[1]
                         )
        else:
            self.node_embedding_receiver = self.node_embedding_sender 

        if edge_encoder is not None:
            self.edge_coding = edge_encoder
        else:
            self.edge_coding = EdgeEncoder(directed=True) # default edge encoder

        self.n_edge_channels = n_atom_basis**2

        self.radial_basis = BesselRBF(cutoff= self.cutoff, n_rbf=self.n_rbf, trainable=self.trainable_rbf)
        self.n_radial_func = self.radial_basis.n_rbf
        self.n_radial_basis = n_radial_basis or self.radial_basis.n_rbf
        self.cutoff_fn = PolynomialCutoff(cutoff=self.cutoff, p= self.cutoff_fn_p)

        # The AngularComponent_GPU version sometimes has trouble with second derivatives
        #if self.device  == torch.device("cpu"):
        #    self.angular_basis = AngularComponent(self.max_l)
        #else:
        #    self.angular_basis = AngularComponent_GPU(self.max_l)
        self.angular_basis = AngularComponent(self.max_l)
        radial_transform = SharedRadialLinearTransform(
                                max_l=self.max_l,
                                radial_dim=self.n_radial_func,
                                radial_embedding_dim=self.n_radial_basis,
                                channel_dim=self.n_edge_channels
                                )
        self.radial_transform = radial_transform
        #self.radial_transform = torch.jit.script(radial_transform)

        self.l_list = self.angular_basis.get_lxlylz_list()
        self.symmetrizer = Symmetrizer(self.max_nu, self.max_l, self.l_list)
        # the JIT version seems to be slower
        #symmetrizer = Symmetrizer_JIT(self.max_nu, self.max_l, self.l_list)
        #self.symmetrizer = torch.jit.script(symmetrizer)

        # for message passing layers
        self.num_message_passing = num_message_passing
        n_angular_sym = 1 + sum([len(self.symmetrizer.vec_dict_allnu[nu]) for nu in range(2, self.max_nu + 1)])
        flat_dim = self.n_radial_basis * n_angular_sym * self.n_edge_channels * (self.num_message_passing + 1)

        self.message_passing_list = nn.ModuleList([
            nn.ModuleList([
                NodeMemory(
                    max_l=self.max_l,
                    radial_embedding_dim=self.n_radial_basis,
                    channel_dim=self.n_edge_channels,
                    **args_message_passing["M"] if "M" in args_message_passing else {}
                    ) if "M" in type_message_passing else None,

                MessageAr(
                    cutoff=cutoff,
                    max_l=self.max_l,
                    radial_embedding_dim=self.n_radial_basis,
                    channel_dim=self.n_edge_channels,
                    **args_message_passing["Ar"] if "Ar" in args_message_passing else {}
                    ) if "Ar" in type_message_passing else None,

                MessageBchi(
                    lxlylz_index = self.angular_basis.get_lxlylz_index(),
                    n_in = self.n_radial_basis * n_angular_sym * self.n_edge_channels,
                    channel_dim = self.n_edge_channels,
                    num_message_passing = self.num_message_passing,
                    # n_out = self.n_edge_channels,
                    **args_message_passing["Bchi"] if "Bchi" in args_message_passing else {}
                    ) if "Bchi" in type_message_passing else None,
            ]) 
            for _ in range(self.num_message_passing)
            ])
        
        
        
        
        if self.readout_type == "weighted_mlp": 
            # Warning: This can introduce a large number of parameters
            self.final_layer = WeightedReadOut_norepeat(
                in_feats=flat_dim,
                dims=atomwise_hidden,
                num_targets=num_targets,  # type: ignore
            )

            self.latent_charge_readout = WeightedReadOut_norepeat(
                in_feats=flat_dim,
                dims=latent_charge_hidden,
                num_targets=num_targets,  # type: ignore
            )
            
            # nn.Sequential(
            #     nn.Linear(flat_dim, atomwise_hidden[0] * 2), nn.SiLU(),
            #     WeightedReadOut(
            #         in_feats=atomwise_hidden[0] * 2,
            #         dims=atomwise_hidden,
            #         num_targets=num_targets,  # type: ignore
            #     ),
                
            # )

            # self.latent_charge_readout = nn.Sequential(
            #     nn.Linear(flat_dim, atomwise_hidden[0] * 2), nn.SiLU(),
            #     WeightedReadOut(
            #         in_feats=atomwise_hidden[0] * 2,
            #         dims=atomwise_hidden,
            #         num_targets=num_targets,  # type: ignore
            #     ),
            # )
            
        elif self.readout_type == "mlp":
            self.final_layer = MLPReadOut(
                    in_feats=flat_dim,
                    dims=atomwise_hidden,
                    num_targets=num_targets,  # type: ignore
                )

            self.latent_charge_readout = MLPReadOut(
                    in_feats=flat_dim,
                    dims=latent_charge_hidden,
                    num_targets=num_targets,  # type: ignore
                )
        
        
        # self.device = device

    def forward(
        self, 
        g: dgl.DGLGraph, state_attr: torch.Tensor | None = None, **kwargs
        # data: Dict[str, torch.Tensor]
    ):
        # setup
        g.ndata["batch"] = torch.repeat_interleave(torch.arange(g.batch_size, device=g.device), g.batch_num_nodes(), dim=0)

        n_nodes = g.ndata['pos'].shape[0]
        if g.ndata["batch"] == None:
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=g.device)
        else:
            batch_now = g.ndata["batch"]

        node_feats_list = []
        node_feats_A_list = []

        # Embeddings
        ## code each node/element in one-hot way
        node_one_hot = self.node_onehot(g.ndata['node_type'] + 1) # +1 to account for the fact that the atomic number is 1-indexed

        ## embed to a different dimension
        node_embedded_sender = self.node_embedding_sender(node_one_hot)
        node_embedded_receiver = self.node_embedding_receiver(node_one_hot)
        ## get the edge type
        src, dst = g.edges()
        src = src.to(torch.int64)
        dst = dst.to(torch.int64)
        edge_index = torch.stack([src, dst], dim=0)

        encoded_edges = self.edge_coding(edge_index=edge_index,
                                         node_type=node_embedded_sender,
                                         node_type_2=node_embedded_receiver,
                                         data=g)

        # compute angular and radial terms
        edge_vectors, edge_lengths = get_edge_vectors_and_lengths(
            positions=g.ndata['pos'],
            edge_index=edge_index,
            shifts=g.edata['pbc_offshift'], # offset in fractional coordinates, offshift in Cartesian coordinates
            normalize=True,
            )
        radial_component = self.radial_basis(edge_lengths) 
        radial_cutoff = self.cutoff_fn(edge_lengths)
        angular_component = self.angular_basis(edge_vectors)

        # combine
        # 4-dimensional tensor: [n_edges, radial_dim, angular_dim, embedding_dim]
        edge_attri = elementwise_multiply_3tensors(
                      radial_component * radial_cutoff,
                      angular_component,
                      encoded_edges
        )

        # sum over edge features to each node
        # 4-dimensional tensor: [n_nodes, radial_dim, angular_dim, embedding_dim]
        node_feat_A = scatter_sum(src=edge_attri, 
                                  index=edge_index[1], 
                                  dim=0, 
                                  dim_size=n_nodes)

        # mix the different radial components
        node_feat_A = self.radial_transform(node_feat_A)
        if hasattr(self, "keep_node_features_A") and self.keep_node_features_A:
            node_feats_A_list.append(node_feat_A)

        # symmetrized B basis
        node_feat_B = self.symmetrizer(node_attr=node_feat_A)
        node_feats_list.append(node_feat_B)

        # message passing
        for nm, mp_Ar, mp_Bchi in self.message_passing_list: 
            if nm is not None:
                momeory_now = nm(node_feat=node_feat_A)
            else:
                momeory_now = 0.0

            if mp_Bchi is not None:
                message_Bchi = mp_Bchi(node_feat=node_feat_B,
                    edge_attri=edge_attri,
                    edge_index=edge_index,
                    )
                node_feat_A_Bchi = scatter_sum(src=message_Bchi,
                                       index=edge_index[1],
                                       dim=0,
                                       dim_size=n_nodes)
                # mix the different radial components
                node_feat_A_Bchi = self.radial_transform(node_feat_A_Bchi)
            else:
                node_feat_A_Bchi = 0.0 

            if mp_Ar is not None:
                message_Ar = mp_Ar(node_feat=node_feat_A,
                    edge_lengths=edge_lengths,
                    radial_cutoff_fn=radial_cutoff,
                    edge_index=edge_index,
                    )

                node_feat_Ar = scatter_sum(src=message_Ar,
                                  index=edge_index[1],
                                  dim=0,
                                  dim_size=n_nodes)
            else:
                node_feat_Ar = 0.0
 
            node_feat_A = node_feat_Ar + node_feat_A_Bchi 
            node_feat_A *= self.mp_norm_factor
            node_feat_A += momeory_now
            if hasattr(self, "keep_node_features_A") and self.keep_node_features_A:
                node_feats_A_list.append(node_feat_A)
            node_feat_B = self.symmetrizer(node_attr=node_feat_A)
            node_feats_list.append(node_feat_B)
     
        node_feats_out = torch.stack(node_feats_list, dim=-1)
        if hasattr(self, "keep_node_features_A") and self.keep_node_features_A:
            node_feats_A_out = torch.stack(node_feats_A_list, dim=-1)
            g.ndata["node_feats_A_out"] = node_feats_A_out
        else:
            node_feats_A_out = None

        # print("Before reshape", node_feats_out.shape)
        node_feats_out = node_feats_out.reshape(node_feats_out.shape[0], -1)
        # print("After reshape", node_feats_out.shape)
        g.ndata["node_feat"] = node_feats_out

        g.ndata["latent_charge"] =  self.latent_charge_readout(g)
        g.ndata["atomic_properties"] = self.final_layer(g)

        
        # print(node_feats_out)

        # print(g.ndata["node_feat"].shape)

        try:
            displacement = g.ndata["displacement"]
        except:
            displacement = None

        # output = {
        #     "positions": g.ndata["pos"],
        #     "cell": g.ndata["cell"],
        #     "displacement": displacement,
        #     "batch": batch_now,
        #     "node_feats": node_feats_out,
        #     #"node_feats_A": node_feats_A_out,
        #     }

        # if hasattr(self, "forward_features") and len(self.forward_features) > 0:
        #     for key in self.forward_features:
        #         if key in data:
        #             output[key] = data[key]
 
        output = dgl.readout_nodes(g, "atomic_properties", op="sum")
        return torch.squeeze(output)
    
    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ):
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes
            graph_converter: Object that implements a get_graph_from_structure.

        Returns:
            output (torch.tensor): output property
        """
        if graph_converter is None:
            from matgl.ext.pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore
        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
        g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        return self(g=g, state_attr=state_feats).detach()

