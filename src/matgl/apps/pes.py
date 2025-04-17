"""Implementation of Interatomic Potentials."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.autograd import grad

import matgl
from matgl.layers import AtomRef, NuclearRepulsion
from matgl.utils.io import IOMixIn

from les.module import Ewald
from les.module import BEC

if TYPE_CHECKING:
    import dgl
    import numpy as np

class Potential(nn.Module, IOMixIn):
    """A class representing an interatomic potential."""

    __version__ = 3

    def __init__(
        self,
        model: nn.Module,
        data_mean: torch.Tensor | float = 0.0,
        data_std: torch.Tensor | float = 1.0,
        element_refs: torch.Tensor | np.ndarray | None = None,
        calc_forces: bool = True,
        calc_stresses: bool = True,
        calc_hessian: bool = False,
        calc_magmom: bool = False,
        calc_repuls: bool = False,
        calc_BEC: bool = False,
        les_sigma: float = 1.0,
        les_dl: float = 2.0,
        les_remove_mean: bool = True,
        les_epsilon_factor: float = 1.0,
        zbl_trainable: bool = False,
        debug_mode: bool = False,
        return_charge: bool = False,
    ):
        """Initialize Potential from a model and elemental references.

        Args:
            model: Model for predicting energies.
            data_mean: Mean of target.
            data_std: Std dev of target.
            element_refs: Element reference values for each element.
            calc_forces: Enable force calculations.
            calc_stresses: Enable stress calculations.
            calc_hessian: Enable hessian calculations.
            calc_magmom: Enable site-wise property calculation.
            calc_repuls: Whether the ZBL repulsion is included
            zbl_trainable: Whether zbl repulsion is trainable
            debug_mode: Return gradient of total energy with respect to atomic positions and lattices for checking
        """
        super().__init__()
        self.save_args(locals())
        self.model = model
        self.calc_forces = calc_forces
        self.calc_stresses = calc_stresses
        self.calc_hessian = calc_hessian
        self.calc_BEC = calc_BEC
        self.calc_magmom = calc_magmom
        self.element_refs: AtomRef | None
        self.debug_mode = debug_mode
        self.calc_repuls = calc_repuls
        self.les_sigma = les_sigma
        self.les_dl = les_dl
        self.les_remove_mean = les_remove_mean
        self.les_epsilon_factor = les_epsilon_factor
        self.return_charge = return_charge

        if calc_repuls:
            self.repuls = NuclearRepulsion(self.model.cutoff, trainable=zbl_trainable)

        if element_refs is not None:
            if not isinstance(element_refs, torch.Tensor):
                element_refs = torch.tensor(element_refs, dtype=matgl.float_th)
            self.element_refs = AtomRef(property_offset=element_refs)
        else:
            self.element_refs = None
        # for backward compatibility
        if data_mean is None:
            data_mean = 0.0
        if not isinstance(data_mean, torch.Tensor):
            data_mean = torch.tensor(data_mean, dtype=matgl.float_th)
        if not isinstance(data_std, torch.Tensor):
            data_std = torch.tensor(data_std, dtype=matgl.float_th)

        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)

        if self.calc_BEC:
            self.ewald = Ewald(
                sigma=self.les_sigma,  # width of the Gaussian on each atom
                dl=self.les_dl,  # grid resolution
            )

            self.bec = BEC(
                remove_mean=self.les_remove_mean,
                epsilon_factor=self.les_epsilon_factor,
            )
        

    def forward(
        self,
        g: dgl.DGLGraph,
        lat: torch.Tensor,
        state_attr: torch.Tensor | None = None,
        l_g: dgl.DGLGraph | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Args:
            g: DGL graph
            lat: lattice
            state_attr: State attrs
            l_g: Line graph.

        Returns:
            (energies, forces, stresses, hessian) or (energies, forces, stresses, hessian, site-wise properties)
        """
        # st (strain) for stress calculations
        st = lat.new_zeros([g.batch_size, 3, 3])
        if self.calc_stresses:
            st.requires_grad_(True)
        lattice = lat @ (torch.eye(3, device=lat.device) + st)
        g.ndata["batch"] = torch.repeat_interleave(torch.arange(g.batch_size, device=lat.device), g.batch_num_nodes(), dim=0)
        g.edata["lattice"] = torch.repeat_interleave(lattice, g.batch_num_edges(), dim=0)
        g.edata["pbc_offshift"] = (g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
        g.ndata["pos"] = (
            g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lattice, g.batch_num_nodes(), dim=0)
        ).sum(dim=1)
        
        ####### process the forces#######
        if self.calc_forces:
            g.ndata["pos"].requires_grad_(True)

        total_energies = self.model(g=g, state_attr=state_attr, l_g=l_g)

        # print(g.ndata.keys())
        # print("total_energies (SR):", total_energies)
        # batch = create_batch_indices(lattice, g.ndata["pos"])

        # batch_num_nodes = g._batch_num_nodes['_N']
        # print("batch_num_nodes shape:", batch_num_nodes.shape)


        # print("g.ndata['latent_charge'] shape:", g.ndata["latent_charge"].shape)
        # print("g.ndata['pos'] shape:", g.ndata["pos"].shape)

        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # print("batch:", batch)
        if self.calc_BEC:
            ewald_E = self.ewald(q = g.ndata["latent_charge"], 
                       r = g.ndata["pos"], 
                       cell = lattice, 
                       batch = g.ndata["batch"]
                       )
            
            # print("ewald_E shape:", ewald_E.shape)
            
            ewald_E = torch.squeeze(ewald_E)
            # print("ewald_E:", ewald_E)
            # print("ewald_E shape:", ewald_E.shape)
            # print("total_energies shape:", total_energies.shape)

            # if total_energies.shape != ewald_E.shape:
            #     total_energies = total_energies.unsqueeze(-1)
            # print("total_energies shape:", total_energies.shape)

            total_energies += ewald_E
            # total_energies = torch.squeeze(total_energies) # .squeeze(-1)

            # print("total_energies (SR+Ewald):", total_energies)

        total_energies = self.data_std * total_energies + self.data_mean

        if self.calc_repuls:
            total_energies += self.repuls(self.model.element_types, g)

        if self.element_refs is not None:
            property_offset = torch.squeeze(self.element_refs(g))
            total_energies += property_offset

        forces = torch.zeros(1)
        stresses = torch.zeros(1)
        hessian = torch.zeros(1)

        grad_vars = [g.ndata["pos"], st] if self.calc_stresses else [g.ndata["pos"]]

        if self.calc_forces:
            grads = grad(
                total_energies,
                grad_vars,
                grad_outputs=torch.ones_like(total_energies),
                create_graph=True,
                retain_graph=True,
            )
            forces = -grads[0]

        if self.calc_hessian:
            r = -grads[0].view(-1)
            s = r.size(0)
            hessian = total_energies.new_zeros((s, s))
            for iatom in range(s):
                tmp = grad([r[iatom]], g.ndata["pos"], retain_graph=iatom < s)[0]
                if tmp is not None:
                    hessian[iatom] = tmp.view(-1)

        if self.calc_stresses:
            volume = (
                torch.abs(torch.det(lattice.float())).half()
                if matgl.float_th == torch.float16
                else torch.abs(torch.det(lattice))
            )
            sts = -grads[1]
            scale = 1.0 / volume * -160.21766208
            sts = [i * j for i, j in zip(sts, scale, strict=False)] if sts.dim() == 3 else [sts * scale]  # type:ignore[assignment]
            stresses = torch.cat(sts)  # type:ignore[call-overload]


        if self.debug_mode:
            return total_energies, grads[0], grads[1]

        if self.calc_magmom:
            return total_energies, forces, stresses, hessian, g.ndata["magmom"]
        
        if self.return_charge:
            bec = self.bec(
                q = g.ndata["latent_charge"], 
                r = g.ndata["pos"], 
                cell = lattice,
                batch = g.ndata["batch"]
            )
            return total_energies, forces, stresses, hessian, g.ndata["latent_charge"], bec
    
        return total_energies, forces, stresses, hessian
